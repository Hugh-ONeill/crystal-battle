"""Only damage the probe state can REPRODUCE may become item evidence.

The damage-bracket ratio re-evaluates a recorded hit against a synthetic
two-mon State that carries stats, types, abilities, items and weather —
and nothing else. No turn order, no terrain, no status on either side, no
defender HP at the time of the hit, no hit counts, no faint counts. When a
move's damage depends on any of that, the modelling error does not vanish;
it lands in the ratio, whose only output is an item claim.

Found from a Ting-Lu branded Choice Specs in the live webs run whose only
special move is Ruination.
"""
import sys
from pathlib import Path

import poke_engine as pe
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import _measurable_damage


def _rolls(move, attacker, defender):
    state = pe.State(
        side_one=pe.Side(pokemon=[attacker] + [
            pe.Pokemon.create_fainted() for _ in range(5)]),
        side_two=pe.Side(pokemon=[defender] + [
            pe.Pokemon.create_fainted() for _ in range(5)]),
    )
    return pe.calculate_damage(state, move, "splash", True)[0]


def _tinglu():
    return pe.Pokemon(
        id="tinglu", level=100, hp=414, maxhp=414, attack=306, defense=306,
        special_attack=140, special_defense=196, speed=140,
        types=("dark", "ground"), base_types=("dark", "ground"),
        ability="vesselofruin", base_ability="vesselofruin",
        item="leftovers", weight_kg=699.7)


def _ursaluna(status="none"):
    return pe.Pokemon(
        id="ursaluna", level=100, hp=425, maxhp=425, attack=372, defense=246,
        special_attack=180, special_defense=196, speed=120,
        types=("ground", "normal"), base_types=("ground", "normal"),
        ability="guts", base_ability="guts", item="flameorb",
        weight_kg=140.0, status=status)


def _ninetales(hp=323):
    return pe.Pokemon(
        id="ninetalesalola", level=100, hp=hp, maxhp=323, attack=140,
        defense=196, special_attack=270, special_defense=236, speed=299,
        types=("ice", "fairy"), base_types=("ice", "fairy"),
        ability="snowwarning", base_ability="snowwarning",
        item="lightclay", weight_kg=19.9)


# ---- the three mechanisms, each shown to produce a false Choice ratio ----

def test_percentage_damage_ratio_balloons_as_the_defender_is_chipped():
    """Ruination: poke-engine models it CORRECTLY, off current HP. Evidence
    is recorded across turns and re-evaluated later, so the same observation
    reads hotter and hotter as our own mon takes chip damage."""
    observed = max(_rolls("ruination", _tinglu(), _ninetales(323)))
    later = max(_rolls("ruination", _tinglu(), _ninetales(80)))
    assert observed / later > 1.38, "premise broken"
    assert not _measurable_damage("ruination")


def test_status_conditional_power_is_excluded():
    """Facade on a Guts user with a Flame Orb: 2x power, 1.5x attack, and
    the probe carries no status at all."""
    clean = max(_rolls("facade", _ursaluna(), _ninetales()))
    real = max(_rolls("facade", _ursaluna("burn"), _ninetales()))
    assert real / clean > 1.38, "premise broken"
    assert not _measurable_damage("facade")


def test_multihit_is_excluded_because_the_calc_returns_one_hit():
    per_hit = max(_rolls("rockblast", _ursaluna(), _ninetales()))
    total = per_hit * 5  # what the protocol reports for a 5-hit roll
    assert total / per_hit > 1.38, "premise broken"
    assert not _measurable_damage("rockblast")


# ---- the classes, by rule ----

@pytest.mark.parametrize("move", [
    "ruination", "superfang", "seismictoss", "nightshade", "endeavor",
    "counter", "mirrorcoat", "metalburst", "finalgambit", "flail",
    "reversal", "gyroball", "electroball", "punishment",
])
def test_fixed_and_variable_power_moves_excluded(move):
    assert not _measurable_damage(move)


@pytest.mark.parametrize("move", [
    "rockblast", "iciclespear", "bulletseed", "populationbomb", "bonerush",
])
def test_multihit_moves_excluded(move):
    assert not _measurable_damage(move)


@pytest.mark.parametrize("move", [
    "facade", "hex", "brine", "venoshock", "acrobatics", "boltbeak",
    "fishiousrend", "payback", "avalanche", "assurance", "storedpower",
    "ragefist", "lastrespects", "eruption", "waterspout", "weatherball",
    "terrainpulse", "expandingforce", "risingvoltage",
    "terablast", "revenge",
])
def test_conditional_power_moves_excluded(move):
    assert not _measurable_damage(move)


def test_random_multiplier_excluded():
    """Fickle Beam doubles on a 30% roll, so no state reproduces it — and a
    2x ratio brands Choice about one hit in three."""
    assert not _measurable_damage("ficklebeam")


@pytest.mark.parametrize("move", ["magnitude", "present", "psywave", "beatup"])
def test_other_random_power_moves_fall_to_the_data_driven_rule(move):
    assert not _measurable_damage(move)


@pytest.mark.parametrize("move", [
    "earthquake", "knockoff", "closecombat", "flamethrower", "icebeam",
    "moonblast", "uturn", "shadowball", "grassknot", "heavyslam",
    "freezedry", "thunderbolt", "dracometeor", "lowkick", "suckerpunch",
    "steelroller", "solarbeam",
])
def test_ordinary_attacks_still_measurable(move):
    """The filter must not swallow the evidence it exists to protect —
    plain base-power moves, and the weight-based ones whose inputs the probe
    genuinely does carry. Sucker Punch and Steel Roller belong here too: their
    condition makes them FAIL, it does not change their power."""
    assert _measurable_damage(move)


def test_unknown_move_is_not_measurable():
    assert not _measurable_damage("notarealmove")


# ---- mixed attackers cannot hold Choice Band or Specs (but CAN hold Scarf) ----

def _obs_using(moves, species="kyurem"):
    """Feed a protocol history where the opponent uses `moves`."""
    from types import SimpleNamespace
    from showdown.set_inference import BattleObservations
    disp = species.replace("ironvaliant", "Iron Valiant").title()
    events = [f"|switch|p2a: {disp}|{disp}|100/100"]
    events += [f"|move|p2a: {disp}|{m}|p1a: X" for m in moves]
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role="p1")
    o = BattleObservations()
    o.update(b)
    return o


def test_attacking_with_both_categories_rules_out_band_and_specs():
    """Kyurem's Loaded Dice build: Icicle Spear physically, Ice Beam and
    Earth Power specially. Each Choice damage item boosts ONE category, so
    half the moveset would be dead weight behind a lock."""
    o = _obs_using(["Icicle Spear", "Ice Beam", "Earth Power"])
    forbidden = o.forbidden("kyurem")
    assert "choiceband" in forbidden
    assert "choicespecs" in forbidden


def test_mixed_attacker_may_still_hold_a_choice_scarf():
    """Scarf boosts SPEED, which every move uses equally — nothing is wasted.
    Iron Valiant (Close Combat + Moonblast) is the canonical mixed Scarf set,
    and is the exact mon whose real Scarf this module exists to catch."""
    o = _obs_using(["Close Combat", "Moonblast"], species="ironvaliant")
    assert "choicescarf" not in o.forbidden("ironvaliant")
    assert "choiceband" in o.forbidden("ironvaliant")


def test_a_special_attacker_with_uturn_is_not_mixed():
    """Pelipper's genuine Choice Specs set is Hurricane / Hydro Pump /
    U-turn. Pivots are carried for the switch, not the stat."""
    o = _obs_using(["Hurricane", "Hydro Pump", "U-turn"], species="pelipper")
    assert "choicespecs" not in o.forbidden("pelipper")


def test_knock_off_on_a_special_attacker_is_not_mixed():
    o = _obs_using(["Shadow Ball", "Knock Off"], species="gholdengo")
    assert "choicespecs" not in o.forbidden("gholdengo")
