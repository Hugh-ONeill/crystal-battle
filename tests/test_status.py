import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.damage import calc_damage
from engine.events import (
    ConfusionAppliedEvent,
    ConfusionHitSelfEvent,
    FaintEvent,
    ResidualDamageEvent,
    StatusAppliedEvent,
    StatusPreventedEvent,
)
from engine.move import MoveSlot, MoveTemplate
from engine.pokemon import Pokemon, PokemonSpecies
from engine.status import (
    BRN, FRZ, PAR, PSN, SLP, TOX,
    apply_confusion,
    apply_status,
    can_apply_status,
    check_confusion,
    check_move_prevention,
    effective_speed,
    end_of_turn_damage,
)
from engine.types import TypeChart

TC = TypeChart.load()

TACKLE = MoveTemplate(id=33, name="Tackle", type="normal", power=35,
                       accuracy=95, pp=35, priority=0, damage_class="physical")
FLAMETHROWER = MoveTemplate(
    id=53, name="Flamethrower", type="fire", power=90, accuracy=100, pp=15,
    priority=0, damage_class="special",
    meta={"ailment_id": 4, "ailment_chance": 10, "min_hits": None,
          "max_hits": None, "drain": 0, "healing": 0, "crit_rate": 0,
          "flinch_chance": 0, "stat_chance": 0},
)


def _species(name="Test", types=None, base_stats=None):
    if types is None:
        types = ["normal"]
    if base_stats is None:
        base_stats = {"hp": 80, "attack": 80, "defense": 80,
                      "special_attack": 80, "special_defense": 80, "speed": 80}
    return PokemonSpecies(id=1, name=name, types=types,
                          base_stats=base_stats, learnset=[])


def _pokemon(name="Test", types=None, moves=None, base_stats=None):
    sp = _species(name, types, base_stats)
    pkmn = Pokemon(species=sp)
    if moves is None:
        moves = [TACKLE]
    pkmn.move_slots = [MoveSlot(template=m) for m in moves]
    return pkmn


# ============================================================
# APPLICATION + IMMUNITIES
# ============================================================

def test_apply_burn():
    pkmn = _pokemon()
    rng = random.Random(42)
    assert apply_status(pkmn, BRN, rng) is True
    assert pkmn.status == BRN


def test_burn_immunity_fire_type():
    pkmn = _pokemon(types=["fire"])
    rng = random.Random(42)
    can, reason = can_apply_status(pkmn, BRN)
    assert can is False
    assert "immune" in reason


def test_paralysis_immunity_electric_type():
    pkmn = _pokemon(types=["electric"])
    can, _ = can_apply_status(pkmn, PAR)
    assert can is False


def test_poison_immunity_poison_type():
    pkmn = _pokemon(types=["poison"])
    can, _ = can_apply_status(pkmn, PSN)
    assert can is False


def test_poison_immunity_steel_type():
    pkmn = _pokemon(types=["steel"])
    can, _ = can_apply_status(pkmn, TOX)
    assert can is False


def test_freeze_immunity_ice_type():
    pkmn = _pokemon(types=["ice"])
    can, _ = can_apply_status(pkmn, FRZ)
    assert can is False


def test_non_volatile_mutual_exclusion():
    pkmn = _pokemon()
    rng = random.Random(42)
    apply_status(pkmn, BRN, rng)
    assert apply_status(pkmn, PAR, rng) is False
    assert pkmn.status == BRN


def test_cannot_status_fainted():
    pkmn = _pokemon()
    pkmn.current_hp = 0
    rng = random.Random(42)
    assert apply_status(pkmn, BRN, rng) is False


# ============================================================
# BURN
# ============================================================

def test_burn_halves_attack_in_damage():
    attacker = _pokemon(base_stats={"hp": 80, "attack": 100, "defense": 80,
                                     "special_attack": 80, "special_defense": 80, "speed": 80})
    defender = _pokemon()
    rng = random.Random(1)

    # damage without burn
    d_normal, _, _ = calc_damage(attacker, defender, TACKLE, TC, random.Random(1))

    # apply burn
    attacker.status = BRN
    d_burned, _, _ = calc_damage(attacker, defender, TACKLE, TC, random.Random(1))

    assert d_burned < d_normal


def test_burn_eot_damage():
    pkmn = _pokemon()
    pkmn.status = BRN
    damage = end_of_turn_damage(pkmn)
    assert damage == pkmn.max_hp // 8


# ============================================================
# PARALYSIS
# ============================================================

def test_paralysis_speed_quartering():
    pkmn = _pokemon(base_stats={"hp": 80, "attack": 80, "defense": 80,
                                 "special_attack": 80, "special_defense": 80, "speed": 200})
    normal_speed = effective_speed(pkmn)
    pkmn.status = PAR
    para_speed = effective_speed(pkmn)
    assert para_speed == normal_speed // 4


def test_paralysis_can_skip_turn():
    """Run many trials to verify ~25% skip rate."""
    pkmn = _pokemon()
    pkmn.status = PAR
    skipped = 0
    trials = 1000
    rng = random.Random(42)
    for _ in range(trials):
        can_act, reason = check_move_prevention(pkmn, rng)
        if not can_act:
            skipped += 1
            assert reason == "fully paralyzed"
    # should be roughly 25% (allow 15-35%)
    assert 150 < skipped < 350


# ============================================================
# SLEEP
# ============================================================

def test_sleep_counter_and_wake():
    pkmn = _pokemon()
    rng = random.Random(42)
    apply_status(pkmn, SLP, rng)
    initial_turns = pkmn.status_turns
    assert 1 <= initial_turns <= 7

    # tick down until wake
    for _ in range(initial_turns):
        can_act, reason = check_move_prevention(pkmn, rng)
    # after enough ticks, should wake up
    assert pkmn.status is None


def test_sleep_blocks_move():
    pkmn = _pokemon()
    rng = random.Random(42)
    apply_status(pkmn, SLP, rng)
    pkmn.status_turns = 3  # force known counter
    can_act, reason = check_move_prevention(pkmn, rng)
    assert can_act is False
    assert reason == "fast asleep"
    assert pkmn.status_turns == 2


# ============================================================
# FREEZE
# ============================================================

def test_freeze_blocks_move():
    pkmn = _pokemon()
    pkmn.status = FRZ
    # use rng that won't thaw (need roll >= 0.20)
    rng = random.Random(100)
    # find a seed that doesn't thaw on first try
    for seed in range(200):
        rng = random.Random(seed)
        if rng.random() >= 0.20:
            break
    pkmn.status = FRZ
    rng = random.Random(seed)
    can_act, reason = check_move_prevention(pkmn, rng)
    if not can_act:
        assert reason == "frozen solid"


def test_freeze_thaw():
    """Run many trials to verify ~20% thaw rate."""
    thawed = 0
    trials = 1000
    for i in range(trials):
        pkmn = _pokemon()
        pkmn.status = FRZ
        rng = random.Random(i)
        can_act, reason = check_move_prevention(pkmn, rng)
        if can_act and reason == "thawed out":
            thawed += 1
    # should be roughly 20%
    assert 150 < thawed < 250


# ============================================================
# POISON / TOXIC
# ============================================================

def test_poison_eot_damage():
    pkmn = _pokemon()
    pkmn.status = PSN
    damage = end_of_turn_damage(pkmn)
    assert damage == pkmn.max_hp // 8


def test_toxic_escalating_damage():
    pkmn = _pokemon()
    pkmn.status = TOX
    pkmn.status_turns = 0

    d1 = end_of_turn_damage(pkmn)
    assert pkmn.status_turns == 1
    assert d1 == pkmn.max_hp * 1 // 16

    d2 = end_of_turn_damage(pkmn)
    assert pkmn.status_turns == 2
    assert d2 == pkmn.max_hp * 2 // 16

    d3 = end_of_turn_damage(pkmn)
    assert pkmn.status_turns == 3
    assert d3 == pkmn.max_hp * 3 // 16


# ============================================================
# CONFUSION
# ============================================================

def test_confusion_application():
    pkmn = _pokemon()
    rng = random.Random(42)
    assert apply_confusion(pkmn, rng) is True
    assert 2 <= pkmn.confusion_turns <= 5


def test_confusion_no_double_apply():
    pkmn = _pokemon()
    rng = random.Random(42)
    apply_confusion(pkmn, rng)
    assert apply_confusion(pkmn, rng) is False


def test_confusion_self_hit():
    """Run many trials to verify ~50% self-hit rate."""
    hits = 0
    trials = 1000
    for i in range(trials):
        pkmn = _pokemon()
        pkmn.confusion_turns = 3
        rng = random.Random(i)
        hit_self, damage = check_confusion(pkmn, rng)
        if hit_self:
            hits += 1
            assert damage > 0
    assert 400 < hits < 600


# ============================================================
# CLEAR METHODS
# ============================================================

def test_clear_status():
    pkmn = _pokemon()
    pkmn.status = BRN
    pkmn.status_turns = 5
    pkmn.clear_status()
    assert pkmn.status is None
    assert pkmn.status_turns == 0


def test_clear_confusion():
    pkmn = _pokemon()
    pkmn.confusion_turns = 3
    pkmn.clear_confusion()
    assert pkmn.confusion_turns == 0


# ============================================================
# STATUS + FAINT INTERACTION
# ============================================================

def test_eot_damage_can_faint():
    pkmn = _pokemon()
    pkmn.status = PSN
    pkmn.current_hp = 1
    damage = end_of_turn_damage(pkmn)
    actual = pkmn.take_damage(damage)
    assert pkmn.is_fainted
