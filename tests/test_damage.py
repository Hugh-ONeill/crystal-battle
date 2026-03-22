import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.damage import calc_damage, calc_expected_damage
from engine.move import MoveTemplate
from engine.pokemon import Pokemon, PokemonSpecies
from engine.types import TypeChart


def _make_species(name, types, base_stats=None):
    if base_stats is None:
        base_stats = {"hp": 80, "attack": 80, "defense": 80,
                      "special_attack": 80, "special_defense": 80, "speed": 80}
    return PokemonSpecies(id=1, name=name, types=types,
                          base_stats=base_stats, learnset=[])


def _make_pokemon(name, types, base_stats=None):
    species = _make_species(name, types, base_stats)
    return Pokemon(species=species)


TACKLE = MoveTemplate(id=33, name="Tackle", type="normal", power=35,
                       accuracy=95, pp=35, priority=0, damage_class="physical")

THUNDERBOLT = MoveTemplate(id=85, name="Thunderbolt", type="electric", power=95,
                            accuracy=100, pp=15, priority=0, damage_class="special")

FLAMETHROWER = MoveTemplate(id=53, name="Flamethrower", type="fire", power=95,
                             accuracy=100, pp=15, priority=0, damage_class="special")

THUNDER_WAVE = MoveTemplate(id=86, name="Thunder Wave", type="electric", power=0,
                             accuracy=100, pp=20, priority=0, damage_class="status")


def test_status_move_no_damage():
    tc = TypeChart.load()
    a = _make_pokemon("Pikachu", ["electric"])
    d = _make_pokemon("Geodude", ["rock", "ground"])
    dmg, eff, crit = calc_damage(a, d, THUNDER_WAVE, tc)
    assert dmg == 0
    assert eff == 1.0
    assert crit is False


def test_damage_is_positive():
    tc = TypeChart.load()
    rng = random.Random(42)
    a = _make_pokemon("Pikachu", ["electric"])
    d = _make_pokemon("Bulbasaur", ["grass", "poison"])
    dmg, eff, crit = calc_damage(a, d, THUNDERBOLT, tc, rng)
    assert dmg > 0


def test_stab_increases_damage():
    tc = TypeChart.load()
    # electric type using electric move vs neutral target
    a_stab = _make_pokemon("Pikachu", ["electric"])
    a_no_stab = _make_pokemon("Pidgey", ["normal", "flying"])
    d = _make_pokemon("Bulbasaur", ["grass", "poison"])

    rng1 = random.Random(42)
    rng2 = random.Random(42)
    dmg_stab, _, _ = calc_damage(a_stab, d, THUNDERBOLT, tc, rng1)
    dmg_no_stab, _, _ = calc_damage(a_no_stab, d, THUNDERBOLT, tc, rng2)
    # stab should do more (crits may differ but on average stab > no stab)
    # use expected damage for deterministic comparison
    exp_stab = calc_expected_damage(a_stab, d, THUNDERBOLT, tc)
    exp_no_stab = calc_expected_damage(a_no_stab, d, THUNDERBOLT, tc)
    assert exp_stab > exp_no_stab


def test_super_effective():
    tc = TypeChart.load()
    a = _make_pokemon("Charmander", ["fire"])
    d = _make_pokemon("Bulbasaur", ["grass", "poison"])
    rng = random.Random(42)
    dmg, eff, _ = calc_damage(a, d, FLAMETHROWER, tc, rng)
    assert eff == 2.0
    assert dmg > 0


def test_immune():
    tc = TypeChart.load()
    a = _make_pokemon("Pikachu", ["electric"])
    d = _make_pokemon("Geodude", ["rock", "ground"])
    rng = random.Random(42)
    dmg, eff, _ = calc_damage(a, d, THUNDERBOLT, tc, rng)
    assert eff == 0.0
    assert dmg == 0


def test_expected_damage_status():
    tc = TypeChart.load()
    a = _make_pokemon("Pikachu", ["electric"])
    d = _make_pokemon("Geodude", ["rock", "ground"])
    assert calc_expected_damage(a, d, THUNDER_WAVE, tc) == 0.0


def test_damage_integer_only():
    """Damage should always be an integer."""
    tc = TypeChart.load()
    a = _make_pokemon("Test", ["normal"])
    d = _make_pokemon("Test2", ["normal"])
    for seed in range(100):
        rng = random.Random(seed)
        dmg, _, _ = calc_damage(a, d, TACKLE, tc, rng)
        assert isinstance(dmg, int)
