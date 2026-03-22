# reward computation with selectable modes
# "shaped": HP diff delta + faint bonus + action bonuses (default)
# "sparse": win/loss only (+1/-1 terminal, 0 otherwise)
# "blended": terminal +1/-1 + HP diff delta * 0.02

from __future__ import annotations

from engine.events import FaintEvent
from engine.player_state import PlayerState
from engine.pokemon import Pokemon
from engine.types import TypeChart

FAINT_REWARD = 0.08
SWITCH_MATCHUP_COEF = 0.0  # disabled: destabilized training (high KL, entropy collapse)

# bonuses for strategic moves (only awarded if the mon survives the turn)
# disabled: hurt critic convergence without meaningfully changing behavior
STATUS_BONUS = 0.0
SETUP_BONUS = 0.0


def _matchup_score(mon: Pokemon, opp: Pokemon, type_chart: TypeChart) -> float:
    """Offensive advantage minus defensive vulnerability for a matchup.

    Returns a value roughly in [-3, 3] range.
    """
    best_off = 0.0
    for slot in mon.move_slots:
        if slot.template.power > 0:
            eff = type_chart.combined_effectiveness(slot.template.type, opp.types)
            best_off = max(best_off, eff)
    best_def = 0.0
    for slot in opp.move_slots:
        if slot.template.power > 0:
            eff = type_chart.combined_effectiveness(slot.template.type, mon.types)
            best_def = max(best_def, eff)
    return best_off - best_def


def compute_reward(
    my_state: PlayerState,
    opp_state: PlayerState,
    prev_hp_diff: float,
    winner: int | None,
    is_p1: bool,
    truncated: bool,
    events: list | None = None,
    opp_player: int = 2,
    switch_matchup_delta: float = 0.0,
    action_type: str = "",
    mode: str = "shaped",
) -> tuple[float, float]:
    """
    Compute reward and return (reward, new_hp_diff).

    Modes:
      shaped:  HP diff delta * 0.1 + faint bonus + action bonuses + terminal +1/-1
      sparse:  terminal +1/-1 only, 0 otherwise
      blended: terminal +1/-1 + HP diff delta * 0.02 (1/5 of shaped weight)
    """
    my_hp = my_state.total_hp_frac
    opp_hp = opp_state.total_hp_frac
    hp_diff = my_hp - opp_hp

    reward = 0.0

    if winner is not None:
        my_num = 1 if is_p1 else 2
        if winner == my_num:
            reward = 1.0
        else:
            reward = -1.0
    elif truncated:
        reward = -0.1
    elif mode == "shaped":
        reward = (hp_diff - prev_hp_diff) * 0.1
        if events:
            my_num = 1 if is_p1 else 2
            opp_num = opp_player
            for e in events:
                if isinstance(e, FaintEvent) and e.player == opp_num:
                    reward += FAINT_REWARD
        # reward voluntary switches that improve type matchup
        if action_type == "switch" and switch_matchup_delta != 0.0:
            reward += switch_matchup_delta * SWITCH_MATCHUP_COEF
        # offset HP-loss penalty for strategic moves (only if the mon survived)
        if not my_state.active.is_fainted:
            if action_type == "status":
                reward += STATUS_BONUS
            elif action_type == "setup":
                reward += SETUP_BONUS
    elif mode == "blended":
        reward = (hp_diff - prev_hp_diff) * 0.02
    # mode == "sparse": reward stays 0.0 for non-terminal steps

    return reward, hp_diff
