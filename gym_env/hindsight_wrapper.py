# hindsight experience replay: relabel rewards after each episode
# based on what actually happened, not heuristic predictions
#
# after a game ends, walks the trajectory backward and adds bonus
# rewards to actions that initiated multi-turn sequences that paid off:
#   - boost move → later KO with boosted stats
#   - status move → cumulative residual damage
#   - voluntary switch → favorable damage trade
#   - sleep move → free uncontested hits

from __future__ import annotations

import gymnasium
import numpy as np


# bonus magnitudes (tuned to be small relative to existing rewards)
BOOST_KO_BONUS = 0.15       # boost move that led to a KO
BOOST_DAMAGE_BONUS = 0.08   # boost move that led to increased damage
STATUS_DRAIN_BONUS = 0.10   # status move where residual damage > 20% HP
SLEEP_FREE_HIT_BONUS = 0.06 # per free hit while target slept
SWITCH_TRADE_BONUS = 0.05   # switch that led to positive HP trade


class HindsightRewardWrapper(gymnasium.Wrapper):
    """Wraps a battle env to relabel rewards with hindsight credit assignment.

    Buffers each episode's trajectory. When the episode ends, walks backward
    to find multi-turn sequences that paid off, and adds bonus rewards to
    the initiating actions. The modified rewards are what PPO sees.
    """

    def __init__(self, env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        self._trajectory = []  # list of (action_type, reward, info_snapshot)
        self._pending_bonus = 0.0

    def reset(self, **kwargs):
        self._trajectory = []
        self._pending_bonus = 0.0
        return super().reset(**kwargs)

    def step(self, action: int):
        # snapshot pre-step state
        battle = self.env.unwrapped._battle
        pre_snapshot = None
        if self.enabled and battle is not None and not battle.is_over:
            pre_snapshot = self._snapshot(battle, action)

        obs, reward, terminated, truncated, info = super().step(action)

        if not self.enabled:
            return obs, reward, terminated, truncated, info

        # add any pending bonus from previous hindsight analysis
        reward += self._pending_bonus
        self._pending_bonus = 0.0

        # record step in trajectory
        if pre_snapshot is not None:
            post_battle = self.env.unwrapped._battle
            post_snapshot = self._post_snapshot(post_battle) if post_battle else {}
            self._trajectory.append({
                "action": action,
                "action_type": info.get("action_type", ""),
                "reward": reward,
                "pre": pre_snapshot,
                "post": post_snapshot,
            })

        # on episode end, relabel rewards with hindsight
        done = terminated or truncated
        if done and len(self._trajectory) >= 2:
            bonuses = self._compute_hindsight_bonuses()
            # apply bonuses to future reward (can't modify past rewards in SB3)
            # instead, we'll add the total bonus to the terminal reward
            total_bonus = sum(bonuses.values())
            reward += total_bonus
            self._trajectory = []

        return obs, reward, terminated, truncated, info

    def _snapshot(self, battle, action):
        """Capture pre-step state for hindsight analysis."""
        p1 = battle.p1
        p2 = battle.p2
        active = p1.active

        snap = {
            "p1_hp_frac": p1.total_hp_frac,
            "p2_hp_frac": p2.total_hp_frac,
            "p1_active_hp": active.hp_frac,
            "p1_active_name": active.name,
            "p2_active_hp": p2.active.hp_frac,
            "p2_active_name": p2.active.name,
            "p1_atk_stage": active.stat_stages.get("attack", 0),
            "p1_spa_stage": active.stat_stages.get("special_attack", 0),
            "p1_alive": p1.alive_count,
            "p2_alive": p2.alive_count,
            "p2_status": p2.active.status,
        }

        # identify action type
        if action < 4 and action < len(active.move_slots):
            move = active.move_slots[action].template
            snap["move_name"] = move.name
            snap["move_power"] = move.power
            snap["move_class"] = move.damage_class
            snap["is_boost"] = move.damage_class == "status" and move.name in _BOOST_MOVES
            snap["is_status_attack"] = move.damage_class == "status" and move.name in _STATUS_MOVES
            snap["is_sleep"] = move.name in _SLEEP_MOVES
        elif action >= 4:
            snap["is_switch"] = True
        else:
            snap["is_boost"] = False
            snap["is_status_attack"] = False

        return snap

    def _post_snapshot(self, battle):
        """Capture post-step state."""
        return {
            "p1_hp_frac": battle.p1.total_hp_frac,
            "p2_hp_frac": battle.p2.total_hp_frac,
            "p1_active_hp": battle.p1.active.hp_frac,
            "p2_active_hp": battle.p2.active.hp_frac,
            "p2_active_name": battle.p2.active.name,
            "p1_atk_stage": battle.p1.active.stat_stages.get("attack", 0),
            "p1_spa_stage": battle.p1.active.stat_stages.get("special_attack", 0),
            "p1_alive": battle.p1.alive_count,
            "p2_alive": battle.p2.alive_count,
            "p2_status": battle.p2.active.status,
        }

    def _compute_hindsight_bonuses(self):
        """Walk the trajectory and find multi-turn sequences that paid off."""
        bonuses = {}
        traj = self._trajectory

        for i, step in enumerate(traj):
            pre = step["pre"]

            # ---- Boost → KO credit ----
            if pre.get("is_boost"):
                bonus = self._check_boost_payoff(traj, i)
                if bonus > 0:
                    bonuses[i] = bonuses.get(i, 0) + bonus

            # ---- Status → drain credit ----
            if pre.get("is_status_attack"):
                bonus = self._check_status_payoff(traj, i)
                if bonus > 0:
                    bonuses[i] = bonuses.get(i, 0) + bonus

            # ---- Sleep → free hits credit ----
            if pre.get("is_sleep"):
                bonus = self._check_sleep_payoff(traj, i)
                if bonus > 0:
                    bonuses[i] = bonuses.get(i, 0) + bonus

            # ---- Switch → trade credit ----
            if pre.get("is_switch"):
                bonus = self._check_switch_payoff(traj, i)
                if bonus > 0:
                    bonuses[i] = bonuses.get(i, 0) + bonus

        return bonuses

    def _check_boost_payoff(self, traj, boost_idx):
        """Check if a boost move led to KOs or increased damage."""
        pre = traj[boost_idx]["pre"]
        active_name = pre["p1_active_name"]
        atk_before = pre["p1_atk_stage"]
        spa_before = pre["p1_spa_stage"]

        bonus = 0.0
        kos_after = 0

        # look ahead up to 8 turns for payoff
        for j in range(boost_idx + 1, min(boost_idx + 8, len(traj))):
            step_j = traj[j]
            # if we switched out, the boost is gone
            if step_j["pre"].get("p1_active_name") != active_name:
                break
            # check for KOs
            p2_alive_before = step_j["pre"]["p2_alive"]
            p2_alive_after = step_j["post"]["p2_alive"]
            if p2_alive_after < p2_alive_before:
                kos_after += 1

        if kos_after >= 2:
            bonus = BOOST_KO_BONUS  # boost led to a sweep
        elif kos_after == 1:
            bonus = BOOST_DAMAGE_BONUS  # boost led to at least one KO

        return bonus

    def _check_status_payoff(self, traj, status_idx):
        """Check if a status move led to significant residual damage."""
        pre = traj[status_idx]["pre"]
        target_name = pre["p2_active_name"]
        target_hp_at_status = pre["p2_active_hp"]

        # track cumulative HP loss on the target
        total_drain = 0.0
        for j in range(status_idx + 1, min(status_idx + 12, len(traj))):
            step_j = traj[j]
            # if opponent switched out, stop tracking
            if step_j["pre"]["p2_active_name"] != target_name:
                break
            # HP difference from residual (not direct damage)
            pre_hp = step_j["pre"]["p2_active_hp"]
            post_hp = step_j["post"]["p2_active_hp"]
            hp_loss = pre_hp - post_hp
            # only count if our action wasn't a damaging move
            if step_j["pre"].get("move_power", 0) == 0 and hp_loss > 0:
                total_drain += hp_loss

        if total_drain > 0.20:
            return STATUS_DRAIN_BONUS
        return 0.0

    def _check_sleep_payoff(self, traj, sleep_idx):
        """Check if a sleep move gave free hits."""
        free_hits = 0
        for j in range(sleep_idx + 1, min(sleep_idx + 5, len(traj))):
            step_j = traj[j]
            post = step_j["post"]
            pre_j = step_j["pre"]
            # free hit: we attacked and opponent couldn't respond
            if pre_j.get("move_power", 0) > 0:
                p2_hp_loss = pre_j["p2_active_hp"] - post["p2_active_hp"]
                p1_hp_loss = pre_j["p1_active_hp"] - post.get("p1_active_hp", pre_j["p1_active_hp"])
                if p2_hp_loss > 0 and p1_hp_loss == 0:
                    free_hits += 1

        return free_hits * SLEEP_FREE_HIT_BONUS

    def _check_switch_payoff(self, traj, switch_idx):
        """Check if a voluntary switch led to a positive HP trade."""
        if switch_idx + 3 >= len(traj):
            return 0.0

        # compare HP differential before switch vs 3 turns later
        pre_switch = traj[switch_idx]["pre"]
        hp_diff_before = pre_switch["p1_hp_frac"] - pre_switch["p2_hp_frac"]

        end_idx = min(switch_idx + 3, len(traj) - 1)
        post = traj[end_idx]["post"]
        hp_diff_after = post["p1_hp_frac"] - post["p2_hp_frac"]

        improvement = hp_diff_after - hp_diff_before
        if improvement > 0.10:
            return SWITCH_TRADE_BONUS

        return 0.0


# move classifications
_BOOST_MOVES = {
    "Swords Dance", "Curse", "Growth", "Agility", "Amnesia",
    "Belly Drum", "Meditate", "Defense Curl", "Harden",
    "Withdraw", "Acid Armor", "Barrier", "Double Team", "Minimize",
    "Sharpen",
}

_STATUS_MOVES = {
    "Toxic", "Thunder Wave", "Leech Seed",
}

_SLEEP_MOVES = {
    "Hypnosis", "Sleep Powder", "Lovely Kiss", "Sing", "Spore",
}
