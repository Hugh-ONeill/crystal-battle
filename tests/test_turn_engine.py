import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.actions import Forfeit, Struggle, Switch, UseMove
from engine.battle_state import BattleState
from engine.events import (
    ConfusionHitSelfEvent,
    FaintEvent,
    MoveEvent,
    ResidualDamageEvent,
    StatusAppliedEvent,
    StatusPreventedEvent,
    SwitchEvent,
    StruggleEvent,
)
from engine.move import MoveSlot, MoveTemplate
from engine.player_state import PlayerState
from engine.pokemon import Pokemon, PokemonSpecies
from engine.status import BRN, PAR, PSN, SLP
from engine.turn_engine import resolve_turn
from engine.types import TypeChart

TC = TypeChart.load()

TACKLE = MoveTemplate(id=33, name="Tackle", type="normal", power=35,
                       accuracy=95, pp=35, priority=0, damage_class="physical")
QUICK_ATTACK = MoveTemplate(id=98, name="Quick Attack", type="normal", power=40,
                              accuracy=100, pp=30, priority=1, damage_class="physical")
THUNDERBOLT = MoveTemplate(id=85, name="Thunderbolt", type="electric", power=95,
                            accuracy=100, pp=15, priority=0, damage_class="special")


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


def _make_battle(team1=None, team2=None, seed=42):
    if team1 is None:
        team1 = [_pokemon(f"P1_{i}") for i in range(3)]
    if team2 is None:
        team2 = [_pokemon(f"P2_{i}") for i in range(3)]
    return BattleState(
        p1=PlayerState(team=team1),
        p2=PlayerState(team=team2),
        rng=random.Random(seed),
    )


def test_basic_move():
    state = _make_battle()
    hp_before = state.p2.active.current_hp
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    # both should have used tackle
    move_events = [e for e in events if isinstance(e, MoveEvent)]
    assert len(move_events) == 2
    assert state.p2.active.current_hp < hp_before or state.p1.active.current_hp < hp_before


def test_switch():
    state = _make_battle()
    original_name = state.p1.active.name
    events = resolve_turn(state, Switch(1), UseMove(0), TC)
    switch_events = [e for e in events if isinstance(e, SwitchEvent)]
    assert len(switch_events) == 1
    assert switch_events[0].player == 1
    assert state.p1.active.name != original_name


def test_switch_before_move():
    """Switches should execute before moves."""
    state = _make_battle()
    events = resolve_turn(state, Switch(1), UseMove(0), TC)
    # first event should be the switch
    assert isinstance(events[0], SwitchEvent)
    assert events[0].player == 1


def test_priority_move():
    """Higher priority moves go first."""
    fast = _pokemon("Fast", moves=[QUICK_ATTACK],
                    base_stats={"hp": 80, "attack": 80, "defense": 80,
                                "special_attack": 80, "special_defense": 80, "speed": 10})
    slow = _pokemon("Slow", moves=[TACKLE],
                    base_stats={"hp": 80, "attack": 80, "defense": 80,
                                "special_attack": 80, "special_defense": 80, "speed": 200})
    state = _make_battle(team1=[fast, _pokemon()], team2=[slow, _pokemon()])
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    move_events = [e for e in events if isinstance(e, MoveEvent)]
    # fast has priority despite lower speed
    assert move_events[0].pokemon_name == "Fast"


def test_speed_ordering():
    """Faster pokemon moves first when priority is equal."""
    fast = _pokemon("FastGuy",
                    base_stats={"hp": 80, "attack": 80, "defense": 80,
                                "special_attack": 80, "special_defense": 80, "speed": 200})
    slow = _pokemon("SlowGuy",
                    base_stats={"hp": 80, "attack": 80, "defense": 80,
                                "special_attack": 80, "special_defense": 80, "speed": 10})
    state = _make_battle(team1=[fast, _pokemon()], team2=[slow, _pokemon()])
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    move_events = [e for e in events if isinstance(e, MoveEvent)]
    assert move_events[0].pokemon_name == "FastGuy"


def test_forfeit():
    state = _make_battle()
    events = resolve_turn(state, Forfeit(), UseMove(0), TC)
    assert state.winner == 2


def test_faint_sets_must_switch():
    """When a pokemon faints, must_switch should be true."""
    weak = _pokemon("Weak")
    weak.current_hp = 1  # will faint from any hit
    state = _make_battle(team2=[weak, _pokemon("Backup")])
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    faint_events = [e for e in events if isinstance(e, FaintEvent)]
    assert len(faint_events) >= 1
    if any(e.player == 2 for e in faint_events):
        assert state.p2.must_switch or state.p2.is_defeated


def test_pp_decrements():
    state = _make_battle()
    pp_before = state.p1.active.move_slots[0].current_pp
    resolve_turn(state, UseMove(0), UseMove(0), TC)
    assert state.p1.active.move_slots[0].current_pp == pp_before - 1


def test_struggle():
    """When all PP exhausted, Struggle should work."""
    pkmn = _pokemon("Exhausted", moves=[TACKLE])
    pkmn.move_slots[0].current_pp = 0
    state = _make_battle(team1=[pkmn, _pokemon()])
    events = resolve_turn(state, Struggle(), UseMove(0), TC)
    struggle_events = [e for e in events if isinstance(e, StruggleEvent)]
    assert len(struggle_events) == 1


def test_valid_actions_normal():
    pkmn = _pokemon(moves=[TACKLE, THUNDERBOLT])
    state = PlayerState(team=[pkmn, _pokemon("Bench1"), _pokemon("Bench2")])
    actions = state.valid_actions()
    # should have 2 move actions + 2 switch actions
    assert len(actions) == 4


def test_valid_actions_forced_switch():
    fainted = _pokemon("Fainted")
    fainted.current_hp = 0
    state = PlayerState(team=[fainted, _pokemon("Alive1"), _pokemon("Alive2")])
    assert state.must_switch
    actions = state.valid_actions()
    # only switch actions
    assert all(isinstance(a, Switch) for a in actions)
    assert len(actions) == 2


def test_action_mask_shape():
    pkmn = _pokemon(moves=[TACKLE, THUNDERBOLT])
    state = PlayerState(team=[pkmn] + [_pokemon(f"B{i}") for i in range(5)])
    mask = state.valid_action_mask()
    assert len(mask) == 10
    assert mask[0] is True   # tackle
    assert mask[1] is True   # thunderbolt
    assert mask[2] is False  # no slot 2
    assert mask[3] is False  # no slot 3
    # team slots: action 4+i = switch to team[i], active (0) masked
    assert mask[4] is False  # team 0 = active, masked
    assert mask[5] is True   # team 1
    assert mask[6] is True   # team 2
    assert mask[9] is True   # team 5


# ============================================================
# STATUS INTEGRATION
# ============================================================

THUNDER_WAVE = MoveTemplate(
    id=86, name="Thunder Wave", type="electric", power=0, accuracy=100,
    pp=20, priority=0, damage_class="status",
    meta={"ailment_id": 1, "ailment_chance": 0, "min_hits": None,
          "max_hits": None, "drain": 0, "healing": 0, "crit_rate": 0,
          "flinch_chance": 0, "stat_chance": 0},
)
FLAMETHROWER = MoveTemplate(
    id=53, name="Flamethrower", type="fire", power=90, accuracy=100, pp=15,
    priority=0, damage_class="special",
    meta={"ailment_id": 4, "ailment_chance": 10, "min_hits": None,
          "max_hits": None, "drain": 0, "healing": 0, "crit_rate": 0,
          "flinch_chance": 0, "stat_chance": 0},
)


def test_status_move_applies_paralysis():
    """Thunder Wave should apply paralysis to the target."""
    user = _pokemon("User", moves=[THUNDER_WAVE])
    target = _pokemon("Target", moves=[TACKLE])
    state = _make_battle(
        team1=[user, _pokemon()],
        team2=[target, _pokemon()],
    )
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    applied = [e for e in events if isinstance(e, StatusAppliedEvent)]
    assert len(applied) == 1
    assert applied[0].status == PAR
    assert state.p2.active.status == PAR


def test_status_prevention_blocks_move():
    """A sleeping pokemon can't attack."""
    sleeper = _pokemon("Sleeper", moves=[TACKLE])
    sleeper.status = SLP
    sleeper.status_turns = 3
    state = _make_battle(
        team1=[sleeper, _pokemon()],
        team2=[_pokemon(), _pokemon()],
    )
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    prevented = [e for e in events if isinstance(e, StatusPreventedEvent)]
    assert len(prevented) == 1
    assert prevented[0].reason == "fast asleep"


def test_eot_burn_damage():
    """Burned pokemon takes residual damage at end of turn."""
    burned = _pokemon("Burned", moves=[TACKLE])
    burned.status = BRN
    hp_before = burned.current_hp
    state = _make_battle(
        team1=[burned, _pokemon()],
        team2=[_pokemon(), _pokemon()],
    )
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    residual = [e for e in events if isinstance(e, ResidualDamageEvent)]
    assert any(e.pokemon_name == "Burned" for e in residual)
    assert burned.current_hp < hp_before


def test_secondary_burn_from_flamethrower():
    """Flamethrower has 10% burn chance -- verify it can proc."""
    burns = 0
    trials = 200
    for seed in range(trials):
        user = _pokemon("User", moves=[FLAMETHROWER], types=["fire"])
        target = _pokemon("Target", moves=[TACKLE])
        state = _make_battle(
            team1=[user, _pokemon()],
            team2=[target, _pokemon()],
            seed=seed,
        )
        events = resolve_turn(state, UseMove(0), UseMove(0), TC)
        if state.p2.active.status == BRN:
            burns += 1
    # should be roughly 10% (allow 3-25%)
    assert 5 < burns < 50


def test_eot_damage_causes_faint():
    """End-of-turn poison damage should faint a pokemon at low HP."""
    poisoned = _pokemon("Poisoned", moves=[TACKLE])
    poisoned.status = PSN
    poisoned.current_hp = 1
    state = _make_battle(
        team1=[_pokemon(), _pokemon()],
        team2=[poisoned, _pokemon()],
    )
    events = resolve_turn(state, UseMove(0), UseMove(0), TC)
    faints = [e for e in events if isinstance(e, FaintEvent) and e.pokemon_name == "Poisoned"]
    assert len(faints) >= 1
