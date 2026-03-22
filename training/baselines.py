# Baseline agents for evaluation

from __future__ import annotations

import random

from engine.actions import Action, Struggle, Switch, UseMove
from engine.damage import calc_expected_damage
from engine.player_state import PlayerState
from engine.pokemon import Pokemon
from engine.stat_stages import MOVE_STAT_EFFECTS
from engine.status import can_apply_status, effective_speed
from engine.types import TypeChart

# ailment_id -> engine status string
_AILMENT_STATUS = {1: "par", 2: "slp", 3: "frz", 4: "brn", 5: "psn", 6: "tox"}


def _ailment_to_str(aid: int) -> str | None:
    return _AILMENT_STATUS.get(aid)


class RandomAgent:
    """Picks uniformly from valid actions."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, my_state: PlayerState, opp_state: PlayerState) -> Action:
        return self._rng.choice(my_state.valid_actions(opp_state))


class MaxDamageAgent:
    """Always picks the highest expected damage move. Never switches."""

    def __init__(self, type_chart: TypeChart | None = None):
        self._tc = type_chart or TypeChart.load()

    def act(self, my_state: PlayerState, opp_state: PlayerState) -> Action:
        if my_state.must_switch:
            # pick teammate with best type advantage
            best_idx = -1
            best_eff = -1.0
            for i, p in enumerate(my_state.team):
                if i == my_state.active_index or p.is_fainted:
                    continue
                eff = 0.0
                for slot in p.move_slots:
                    if slot.template.power > 0 and slot.has_pp:
                        e = self._tc.combined_effectiveness(
                            slot.template.type, opp_state.active.types
                        )
                        eff = max(eff, e)
                if eff > best_eff:
                    best_eff = eff
                    best_idx = i
            if best_idx >= 0:
                return Switch(team_index=best_idx)
            # fallback: first alive
            for i, p in enumerate(my_state.team):
                if i != my_state.active_index and not p.is_fainted:
                    return Switch(team_index=i)

        if not my_state.active.has_any_pp():
            return Struggle()

        # pick move with highest expected damage
        best_slot = -1
        best_dmg = -1.0
        opp_asleep = opp_state.active.status == "slp"
        for i, slot in enumerate(my_state.active.move_slots):
            if not slot.has_pp or slot.template.power == 0:
                continue
            if slot.template.id == 138 and not opp_asleep:  # Dream Eater
                continue
            dmg = calc_expected_damage(
                my_state.active, opp_state.active,
                slot.template, self._tc,
            )
            if dmg > best_dmg:
                best_dmg = dmg
                best_slot = i

        if best_slot >= 0:
            return UseMove(slot_index=best_slot)
        # no damaging moves available — use first move with PP
        for i, slot in enumerate(my_state.active.move_slots):
            if slot.has_pp:
                return UseMove(slot_index=i)
        return Struggle()


class SmartAgent:
    """Strong heuristic bot: scores all 10 actions and picks the best.

    Considers speed tiers, stat stages, burn/paralysis, weather, screens,
    status immunity, setup timing, recovery, switch-in cost, and 2-turn
    KO lookahead.
    """

    def __init__(self, type_chart: TypeChart | None = None, seed: int | None = None):
        self._tc = type_chart or TypeChart.load()
        self._rng = random.Random(seed)

    # ---- Core damage helpers ----

    def _move_dmg_frac(self, attacker: Pokemon, defender: Pokemon,
                       move: "MoveTemplate",
                       screens: "SideConditions | None" = None) -> float:
        """Expected damage as fraction of defender max HP."""
        dmg = calc_expected_damage(attacker, defender, move, self._tc, screens=screens)
        return dmg / defender.max_hp if defender.max_hp > 0 else 0.0

    def _best_move(self, attacker: Pokemon, defender: Pokemon,
                   screens: "SideConditions | None" = None) -> tuple[int, float]:
        """(slot_index, damage_frac) for strongest damaging move."""
        best_slot = -1
        best_frac = 0.0
        for i, slot in enumerate(attacker.move_slots):
            if not slot.has_pp or slot.template.power == 0:
                continue
            frac = self._move_dmg_frac(attacker, defender, slot.template, screens)
            if frac > best_frac:
                best_frac = frac
                best_slot = i
        return best_slot, best_frac

    def _i_am_faster(self, me: Pokemon, them: Pokemon) -> bool:
        return effective_speed(me) > effective_speed(them)

    # ---- Strategic pattern recognition ----

    def _has_move_named(self, pkmn: Pokemon, *names: str) -> bool:
        return any(s.template.name in names and s.has_pp for s in pkmn.move_slots)

    def _strategic_bonus(self, slot_idx: int, me: Pokemon, them: Pokemon,
                         my_state: PlayerState, opp_state: PlayerState) -> float:
        """Bonus for recognized competitive strategies."""
        mv = me.move_slots[slot_idx].template
        bonus = 0.0

        # ---- Toxic stall: Toxic + Protect/recovery = drain them out ----
        if mv.name == "Toxic" and them.status is None:
            if self._has_move_named(me, "Protect", "Detect", "Recover",
                                    "Softboiled", "Milk Drink", "Rest"):
                bonus += 3.0  # toxic is extra valuable with sustain
        if mv.name in ("Protect", "Detect") and them.status == "tox":
            bonus += 3.0  # stall out toxic damage

        # ---- Para-sweep: paralyze then sweep with slower hard hitter ----
        # only apply if we're not already dealing good damage (don't waste turns)
        if mv.name in ("Thunder Wave", "Stun Spore", "Glare"):
            meta = mv.meta or {}
            aid = meta.get("ailment_id", 0)
            if aid == 1 and them.status is None:
                _, my_best_frac = self._best_move(me, them, opp_state.side)
                if my_best_frac < 0.3:  # only para if we can't hit hard ourselves
                    for p in my_state.team:
                        if p.is_fainted or p is me:
                            continue
                        _, dmg_frac = self._best_move(p, them, opp_state.side)
                        if dmg_frac > 0.4 and effective_speed(p) < effective_speed(them):
                            bonus += 2.0
                            break

        # ---- Sweep after boost: already boosted, go for the kill ----
        if mv.power > 0:
            atk_stage = me.stat_stages.get("attack", 0)
            spa_stage = me.stat_stages.get("special_attack", 0)
            if (mv.damage_class == "physical" and atk_stage >= 2) or \
               (mv.damage_class == "special" and spa_stage >= 2):
                bonus += 3.0  # boosted attack, press the advantage

        # ---- Curselax: Curse + Rest/Sleep Talk ----
        if mv.name == "Curse" and self._has_move_named(me, "Rest", "Sleep Talk"):
            if me.hp_frac > 0.5:
                bonus += 3.0  # CurseLax is a proven win condition

        # ---- Sleep Talk while asleep: don't waste turns ----
        if mv.name == "Sleep Talk" and me.status == "slp":
            bonus += 8.0  # this is the play while sleeping

        # ---- Belly Drum + priority: classic combo ----
        if mv.name == "Belly Drum" and self._has_move_named(me, "Quick Attack",
                                                             "Mach Punch", "ExtremeSpeed"):
            if me.hp_frac >= 0.9 and me.stat_stages.get("attack", 0) == 0:
                bonus += 4.0  # full HP Belly Drum into priority sweep

        # ---- Spikes + phazing: force entry hazard damage ----
        if mv.name in ("Whirlwind", "Roar") and opp_state.side.spikes:
            bonus += 3.0  # phaze into spikes

        # ---- Leech Seed + Protect: SubSeed stalling ----
        if mv.name in ("Protect", "Detect") and them.leech_seeded:
            bonus += 2.0  # drain while safe

        # ---- Endgame: last mon vs last mon, be aggressive ----
        my_alive = sum(1 for p in my_state.team if not p.is_fainted)
        opp_alive = sum(1 for p in opp_state.team if not p.is_fainted)
        if my_alive == 1 and opp_alive == 1 and mv.power > 0:
            bonus += 2.0  # no switching option, commit to attacking

        return bonus

    # ---- Action scoring ----

    def _score_move(self, slot_idx: int, me: Pokemon, them: Pokemon,
                    my_state: PlayerState, opp_state: PlayerState,
                    opp_best_frac: float) -> float:
        """Score a move action. Higher = better."""
        slot = me.move_slots[slot_idx]
        mv = slot.template
        faster = self._i_am_faster(me, them)
        opp_screens = opp_state.side

        # strategic pattern bonus -- scale down if opponent is pure aggro
        # (if opponent only has damaging moves, strategy wastes turns)
        opp_has_status = any(
            s.template.power == 0 and s.has_pp for s in them.move_slots
        )
        strat_scale = 1.0 if opp_has_status else 0.3
        strat_bonus = self._strategic_bonus(slot_idx, me, them, my_state, opp_state) * strat_scale

        # ---- Damaging moves ----
        if mv.power > 0:
            frac = self._move_dmg_frac(me, them, mv, opp_screens)

            # base value: damage fraction
            score = frac * 10.0

            # KO bonus: massive
            if frac >= them.hp_frac:
                score += 20.0
                # priority KO when slower: even better
                if not faster and mv.priority > 0:
                    score += 5.0

            # 2-turn KO: if I survive their hit and can KO next turn
            elif frac * 2 >= them.hp_frac:
                if faster or opp_best_frac < me.hp_frac:
                    score += 3.0

            # faster = I deal damage before they can KO me
            if faster:
                score += 1.0

            # type effectiveness bonus
            eff = self._tc.combined_effectiveness(mv.type, them.types)
            if eff >= 2.0:
                score += 2.0
            elif eff <= 0.5:
                score -= 3.0
            elif eff == 0.0:
                score = -50.0  # immune, never pick

            # accuracy penalty
            acc = mv.accuracy
            if acc is not None and acc < 100:
                score *= acc / 100.0

            # self-KO moves: only if worth it
            if mv.name in ("Explosion", "Self-Destruct", "Selfdestruct"):
                alive = sum(1 for p in my_state.team if not p.is_fainted)
                if alive <= 1:
                    score -= 30.0  # last mon, never suicide
                elif me.hp_frac > 0.5:
                    score -= 15.0  # too healthy to sacrifice
                elif me.hp_frac > 0.25:
                    score -= 5.0  # only when desperate

            return score + strat_bonus

        # ---- Non-damaging moves: compute base then add strat_bonus ----
        base = self._score_nondamaging(mv, me, them, my_state, opp_state,
                                       faster, opp_best_frac)
        return base + strat_bonus

    def _score_nondamaging(self, mv, me, them, my_state, opp_state,
                           faster, opp_best_frac) -> float:
        """Score a non-damaging move. Called from _score_move."""
        meta = mv.meta or {}
        aid = meta.get("ailment_id", 0)

        # ---- Status ailment moves ----
        if aid > 0 and mv.damage_class == "status":
            status_str = _ailment_to_str(aid)
            if status_str:
                can, _ = can_apply_status(them, status_str)
                if not can:
                    return -50.0

            if aid == 2:  # sleep
                return 12.0 if them.hp_frac > 0.3 else 4.0
            if aid == 1:  # paralysis
                if not faster:
                    return 10.0 if them.hp_frac > 0.3 else 3.0
                return 4.0
            if aid in (5, 6) and mv.name == "Toxic":
                return 8.0 if them.hp_frac > 0.5 else 2.0
            if aid == 5:  # poison
                return 5.0 if them.hp_frac > 0.5 else 1.0
            if aid == 4:  # burn
                return 7.0 if them.hp_frac > 0.3 else 2.0
            if aid == 7:  # confusion
                return 4.0 if them.hp_frac > 0.4 else 1.0
            return 3.0

        # ---- Recovery ----
        if mv.name in ("Recover", "Softboiled", "Morning Sun", "Synthesis",
                        "Milk Drink", "Moonlight"):
            if me.hp_frac >= 0.9:
                return -5.0
            if me.hp_frac < 0.3:
                return 11.0
            if me.hp_frac < 0.6:
                return 7.0 if opp_best_frac < 0.5 else 3.0
            return 1.0

        if mv.name == "Rest":
            if me.hp_frac >= 0.8:
                return -5.0
            if me.hp_frac < 0.25:
                return 9.0
            return 4.0 if opp_best_frac < 0.4 else 1.0

        # ---- Setup / stat boosts ----
        if mv.name in MOVE_STAT_EFFECTS:
            effects = MOVE_STAT_EFFECTS[mv.name]
            is_self_boost = any(t == "self" and s > 0 for _, s, t in effects)
            is_opp_debuff = any(t == "opponent" for _, s, t in effects)

            if is_self_boost:
                maxed = all(
                    me.stat_stages.get(stat, 0) >= 4
                    for stat, stages, target in effects
                    if target == "self" and stages > 0
                )
                if maxed:
                    return -5.0
                if me.hp_frac > 0.6 and opp_best_frac < me.hp_frac:
                    return 9.0
                if me.hp_frac > 0.4 and opp_best_frac < 0.3:
                    return 7.0
                return 1.0

            if is_opp_debuff:
                return 4.0 if them.hp_frac > 0.5 else 1.0

        # ---- Screens ----
        if mv.name == "Reflect":
            return -10.0 if my_state.side.reflect_turns > 0 else (6.0 if me.hp_frac > 0.5 else 2.0)
        if mv.name == "Light Screen":
            return -10.0 if my_state.side.light_screen_turns > 0 else (6.0 if me.hp_frac > 0.5 else 2.0)

        # ---- Spikes ----
        if mv.name == "Spikes":
            return -10.0 if opp_state.side.spikes else (5.0 if me.hp_frac > 0.5 else 1.0)

        # ---- Leech Seed ----
        if mv.name == "Leech Seed":
            return -10.0 if them.leech_seeded else 6.0

        # ---- Phazing ----
        if mv.name in ("Whirlwind", "Roar"):
            return 8.0 if any(v > 0 for v in them.stat_stages.values()) else 0.0

        return 0.0

    def _score_switch_to(self, bench_idx: int, me: Pokemon, them: Pokemon,
                         my_state: PlayerState, opp_state: PlayerState,
                         opp_best_frac: float) -> float:
        """Score switching to a bench mon. Higher = better."""
        p = my_state.team[bench_idx]
        if p.is_fainted:
            return -999.0

        # how much damage will the switch-in take? (opponent attacks while we switch)
        _, opp_dmg_vs_switch = self._best_move(them, p, my_state.side)
        if opp_dmg_vs_switch >= p.hp_frac:
            return -10.0  # switching into a KO

        # how good is the new matchup?
        _, my_dmg = self._best_move(p, them, opp_state.side)
        matchup = (my_dmg - opp_dmg_vs_switch) * 10.0

        # speed advantage
        if effective_speed(p) > effective_speed(them):
            matchup += 1.5

        # HP-weighted
        matchup *= p.hp_frac

        # type resistance bonus
        for slot in them.move_slots:
            if slot.template.power > 0:
                eff = self._tc.combined_effectiveness(slot.template.type, p.types)
                if eff <= 0.5:
                    matchup += 2.0
                    break

        # penalty for switching: lose a turn, take a hit
        matchup -= 3.0

        # urgency: if current mon is about to die, switching is cheaper
        if opp_best_frac >= me.hp_frac:
            matchup += 4.0  # we're dead anyway, switch cost is low

        return matchup

    def act(self, my_state: PlayerState, opp_state: PlayerState) -> Action:
        me = my_state.active
        them = opp_state.active

        # forced switch
        if my_state.must_switch:
            return self._pick_best_switch(my_state, opp_state)

        if not me.has_any_pp():
            return Struggle()

        # opponent's best damage against me (for context)
        _, opp_best_frac = self._best_move(them, me, my_state.side)

        # score all actions
        best_score = -999.0
        best_action: Action = Struggle()

        # score moves
        for i, slot in enumerate(me.move_slots):
            if not slot.has_pp:
                continue
            score = self._score_move(i, me, them, my_state, opp_state, opp_best_frac)
            if score > best_score:
                best_score = score
                best_action = UseMove(slot_index=i)

        # score switches
        for i, p in enumerate(my_state.team):
            if i == my_state.active_index or p.is_fainted:
                continue
            score = self._score_switch_to(i, me, them, my_state, opp_state, opp_best_frac)
            if score > best_score:
                best_score = score
                best_action = Switch(team_index=i)

        return best_action

    def _pick_best_switch(self, my_state: PlayerState, opp_state: PlayerState) -> Switch:
        """Forced switch: pick best matchup."""
        them = opp_state.active
        best_idx = -1
        best_score = -999.0
        for i, p in enumerate(my_state.team):
            if i == my_state.active_index or p.is_fainted:
                continue
            _, my_dmg = self._best_move(p, them, opp_state.side)
            _, opp_dmg = self._best_move(them, p, my_state.side)
            score = (my_dmg - opp_dmg) * p.hp_frac
            if effective_speed(p) > effective_speed(them):
                score += 0.5
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            return Switch(team_index=best_idx)
        for i, p in enumerate(my_state.team):
            if i != my_state.active_index and not p.is_fainted:
                return Switch(team_index=i)
        raise RuntimeError("No valid switch target")
