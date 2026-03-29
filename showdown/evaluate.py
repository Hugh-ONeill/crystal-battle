# Gen 2 battle position evaluation function
# hand-crafted heuristic inspired by Foul Play's eval, tuned for GSC OU
#
# returns a score from P1's perspective (positive = P1 advantage)

from __future__ import annotations

from engine.battle_state import BattleState
from engine.player_state import PlayerState
from engine.pokemon import Pokemon
from engine.types import TypeChart
from engine.stat_stages import get_stage_multiplier
from engine.status import BRN, PAR, SLP, FRZ, PSN, TOX

# Gen 2: physical/special split by type
PHYSICAL_TYPES = {"normal", "fighting", "flying", "poison", "ground",
                  "rock", "bug", "ghost", "steel"}
SPECIAL_TYPES = {"fire", "water", "electric", "grass", "ice",
                 "psychic", "dragon", "dark"}


# ============================================================
# SCORING CONSTANTS
# ============================================================

# per-pokemon base values
ALIVE_BONUS = 80         # being alive at all is worth a lot in gen2
HP_VALUE = 100           # full HP = 100 pts, scales linearly with hp_frac

# status penalties (gen2 status is brutal -- no easy switching out)
STATUS_PENALTY = {
    FRZ: -50,   # frozen is devastating, 20% thaw per turn
    SLP: -30,   # sleep is 1-7 turns, very strong in gen2
    TOX: -35,   # toxic ramps fast, rest is the only cure
    PAR: -25,   # 25% full para + speed quartered
    BRN: -25,   # 1/8 HP per turn + halved physical attack
    PSN: -12,   # 1/8 HP per turn, less impactful than toxic
}

# volatile status
CONFUSION_PENALTY = -15
LEECH_SEED_PENALTY = -25   # drains 1/8 HP per turn + heals opponent

# stat boost values (per stage)
# gen2 boosts are very impactful -- curse snorlax, SD marowak, etc.
BOOST_VALUE = {
    "attack": 18,
    "defense": 12,
    "special_attack": 18,
    "special_defense": 12,
    "speed": 20,       # speed control is huge in gen2
    "accuracy": 4,
    "evasion": 8,      # evasion is annoying but not game-winning
}

# diminishing returns multiplier per stage (-6 to +6)
# maps stage -> effective value multiplier
# +1 is worth full value, +6 is worth less per stage
STAGE_MULTIPLIER = {
    -6: -2.5, -5: -2.4, -4: -2.2, -3: -2.0, -2: -1.6, -1: -1.0,
    0: 0.0,
    1: 1.0, 2: 1.6, 3: 2.0, 4: 2.2, 5: 2.4, 6: 2.5,
}

# side conditions
SPIKES_PENALTY = -8      # per alive reserve (they take damage on switch)
REFLECT_BONUS = 20       # halves physical damage
LIGHT_SCREEN_BONUS = 20  # halves special damage

# weather bonuses (per-side, depends on team composition)
SANDSTORM_ROCK_SPDEF_BONUS = 10  # rock types get 1.5x SpDef

# active matchup
TYPE_ADVANTAGE_BONUS = 15   # having a good offensive matchup
SPEED_ADVANTAGE_BONUS = 8   # moving first matters a lot in gen2


# ============================================================
# EVALUATION
# ============================================================

def evaluate(state: BattleState, tc: TypeChart) -> float:
    """Evaluate a battle position from P1's perspective.

    Returns a float score. Positive = P1 advantage, negative = P2 advantage.
    Terminal states return large values.
    """
    if state.is_over:
        if state.winner == 1:
            return 1000.0
        elif state.winner == 2:
            return -1000.0
        return 0.0

    p1_score = _score_side(state.p1, state.p2, state, tc)
    p2_score = _score_side(state.p2, state.p1, state, tc)

    return p1_score - p2_score


def _score_side(player: PlayerState, opponent: PlayerState,
                state: BattleState, tc: TypeChart) -> float:
    """Score one side of the battle."""
    score = 0.0
    reserves_alive = player.alive_count - (0 if player.active.is_fainted else 1)

    # ---- score each pokemon ----
    for mon in player.team:
        score += _score_pokemon(mon)

    # ---- side conditions ----
    side = player.side

    # spikes hurt on every future switch-in, scaled by living reserves
    if side.spikes:
        score += SPIKES_PENALTY * reserves_alive

    if side.reflect_turns > 0:
        score += REFLECT_BONUS
    if side.light_screen_turns > 0:
        score += LIGHT_SCREEN_BONUS

    # ---- active matchup ----
    if not player.active.is_fainted and not opponent.active.is_fainted:
        score += _active_matchup_score(player.active, opponent.active, tc, state)

    # ---- weather effects ----
    if state.weather == "sandstorm":
        for mon in player.team:
            if not mon.is_fainted and "rock" in mon.types:
                score += SANDSTORM_ROCK_SPDEF_BONUS

    return score


def _score_pokemon(mon: Pokemon) -> float:
    """Score a single pokemon's value."""
    if mon.is_fainted:
        return 0.0

    score = ALIVE_BONUS
    score += HP_VALUE * mon.hp_frac

    # status
    if mon.status is not None:
        base_penalty = STATUS_PENALTY.get(mon.status, 0)
        if mon.status == BRN:
            # burn is worse on physical attackers
            if _is_physical_attacker(mon):
                base_penalty = -35
        elif mon.status == SLP:
            # sleep is less bad if they have Sleep Talk
            if _has_move(mon, "Sleep Talk"):
                base_penalty = -12
        elif mon.status == TOX:
            # toxic gets worse over time
            turns = mon.status_turns
            if turns >= 4:
                base_penalty = -45
        score += base_penalty

    # volatiles
    if mon.confusion_turns > 0:
        score += CONFUSION_PENALTY
    if mon.leech_seeded:
        score += LEECH_SEED_PENALTY

    # stat boosts
    for stat, base_val in BOOST_VALUE.items():
        stage = mon.stat_stages.get(stat, 0)
        if stage != 0:
            score += base_val * STAGE_MULTIPLIER.get(stage, 0.0)

    # recharging/locked penalties
    if mon.recharging:
        score -= 15  # wasting a turn
    if mon.locked_move is not None and mon.locked_turns <= 1:
        score -= 10  # about to be confused from lock-in ending

    return score


def _active_matchup_score(my_active: Pokemon, opp_active: Pokemon,
                          tc: TypeChart, state: BattleState) -> float:
    """Score the current active vs active matchup."""
    score = 0.0

    # ---- offensive type coverage ----
    # best move effectiveness against opponent
    best_eff = 0.0
    for slot in my_active.move_slots:
        if slot.has_pp and slot.template.power > 0:
            eff = tc.combined_effectiveness(slot.template.type, opp_active.types)
            # factor in STAB
            if slot.template.type in my_active.types:
                eff *= 1.5
            if eff > best_eff:
                best_eff = eff

    # opponent's best move effectiveness against us
    opp_best_eff = 0.0
    for slot in opp_active.move_slots:
        if slot.has_pp and slot.template.power > 0:
            eff = tc.combined_effectiveness(slot.template.type, my_active.types)
            if slot.template.type in opp_active.types:
                eff *= 1.5
            if eff > opp_best_eff:
                opp_best_eff = eff

    # net matchup advantage
    if best_eff >= 3.0:       # super effective STAB
        score += TYPE_ADVANTAGE_BONUS * 2
    elif best_eff >= 2.0:     # super effective
        score += TYPE_ADVANTAGE_BONUS
    elif best_eff <= 0.5:     # resisted
        score -= TYPE_ADVANTAGE_BONUS * 0.5

    if opp_best_eff >= 3.0:
        score -= TYPE_ADVANTAGE_BONUS * 2
    elif opp_best_eff >= 2.0:
        score -= TYPE_ADVANTAGE_BONUS
    elif opp_best_eff <= 0.5:
        score += TYPE_ADVANTAGE_BONUS * 0.5

    # ---- speed comparison ----
    my_speed = _effective_speed(my_active)
    opp_speed = _effective_speed(opp_active)
    if my_speed > opp_speed:
        score += SPEED_ADVANTAGE_BONUS
    elif opp_speed > my_speed:
        score -= SPEED_ADVANTAGE_BONUS

    return score


def _effective_speed(mon: Pokemon) -> int:
    """Get effective speed with stat stages and paralysis."""
    speed = mon.stats["speed"]
    stage = mon.stat_stages.get("speed", 0)
    if stage != 0:
        num, den = get_stage_multiplier(stage)
        speed = speed * num // den
    if mon.status == PAR:
        speed //= 4
    return speed


def _is_physical_attacker(mon: Pokemon) -> bool:
    """Check if a pokemon primarily uses physical moves."""
    phys_power = 0
    spec_power = 0
    for slot in mon.move_slots:
        if slot.template.power > 0:
            if slot.template.type in PHYSICAL_TYPES:
                phys_power += slot.template.power
            else:
                spec_power += slot.template.power
    return phys_power > spec_power


def _has_move(mon: Pokemon, move_name: str) -> bool:
    """Check if a pokemon has a specific move."""
    return any(slot.template.name == move_name for slot in mon.move_slots)
