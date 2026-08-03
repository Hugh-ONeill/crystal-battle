"""Damage brackets must measure against a MAX-INVESTED attacker, not our
canonical-spread guess.

The brackets (>1.38 Choice, 1.15-1.38 Life Orb) are derived from ITEM
multipliers, so they only mean what they claim when the denominator is the
same mon holding nothing. Measuring against the canonical spread instead
conflates "they hold an item" with "they invested where my guess did not" —
and the second is the ordinary case, since any mon meant to attack invests in
Atk or SpA. In the gen9ou curated corpus, investment alone clears the Choice
bracket for 57.9% of (set, attacking-stat) pairs, and 99.4% of the pairs
where the canonical set does not invest in that category at all.

A false Choice brand is not cosmetic: it asserts a LOCK, so the search plans
against one move the target may switch off freely, and hands it free setup.
"""
import sys
from pathlib import Path

import poke_engine as pe
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import BattleObservations


def _mon(**kw):
    base = dict(id="gliscor", level=100, hp=353, maxhp=353,
                attack=226, defense=299, special_attack=140,
                special_defense=176, speed=222,
                types=("ground", "flying"), base_types=("ground", "flying"),
                ability="poisonheal", base_ability="poisonheal",
                item="toxicorb", weight_kg=42.5)
    base.update(kw)
    return pe.Pokemon(**base)


def _defender():
    return pe.Pokemon(
        id="tinglu", level=100, hp=414, maxhp=414, attack=306,
        defense=306, special_attack=140, special_defense=196, speed=140,
        types=("dark", "ground"), base_types=("dark", "ground"),
        ability="vesselofruin", base_ability="vesselofruin",
        item="leftovers", weight_kg=699.7)


def _obs_with_hit(damage):
    """One clean Knock Off from Gliscor, `damage` HP taken."""
    o = BattleObservations()
    o.damage_evidence.append({
        "species": "gliscor", "move": "knockoff", "our_species": "tinglu",
        "damage": damage, "weather": "none", "se": False,
    })
    return o


def _max_roll(attacker):
    state = pe.State(
        side_one=pe.Side(pokemon=[attacker] + [
            pe.Pokemon.create_fainted() for _ in range(5)]),
        side_two=pe.Side(pokemon=[_defender()] + [
            pe.Pokemon.create_fainted() for _ in range(5)]),
    )
    rolls = pe.calculate_damage(state, "knockoff", "splash", True)[0]
    return max(rolls)


# canonical Gliscor is the defensive Toxic/Protect set: 0 Atk EVs.
# The offensive variant is max-invested — the same mon, still no item.
CANON = _mon(attack=226)
INVESTED = _mon(attack=317, defense=317, special_attack=317)


def test_investment_alone_would_have_read_as_choice():
    """Guards the premise: this really is a >1.38 ratio under the old
    denominator, so the test below is testing something."""
    ratio = _max_roll(INVESTED) / _max_roll(CANON)
    assert ratio > 1.38, f"premise broken: investment only worth {ratio:.2f}x"


def test_max_invested_hit_is_not_branded_choice():
    o = _obs_with_hit(_max_roll(INVESTED))
    got = o.damage_item_upgrade("gliscor", CANON, {"tinglu": _defender()},
                                invested_probe=INVESTED)
    assert got is None, f"branded {got} for plain EV investment"


def test_genuine_choice_band_still_detected():
    """The brackets are unchanged — only the baseline moved. A real Band on a
    max-invested attacker still reads 1.5x and must still be caught."""
    o = _obs_with_hit(int(_max_roll(INVESTED) * 1.5))
    got = o.damage_item_upgrade("gliscor", CANON, {"tinglu": _defender()},
                                invested_probe=INVESTED)
    assert got == "choiceband", got


def test_genuine_life_orb_still_detected():
    o = _obs_with_hit(int(_max_roll(INVESTED) * 1.3))
    got = o.damage_item_upgrade("gliscor", CANON, {"tinglu": _defender()},
                                invested_probe=INVESTED)
    assert got == "lifeorb", got


def test_kill_switch_restores_old_denominator(monkeypatch):
    import showdown.set_inference as si
    monkeypatch.setattr(si, "_INVESTED_DENOM", False)
    o = _obs_with_hit(_max_roll(INVESTED))
    got = o.damage_item_upgrade("gliscor", CANON, {"tinglu": _defender()},
                                invested_probe=INVESTED)
    assert got == "choiceband", f"kill switch did not restore the bug: {got}"


def test_no_probe_supplied_falls_back_to_canonical():
    """Callers that pass no invested probe keep the old behaviour rather than
    silently skipping inference."""
    o = _obs_with_hit(_max_roll(INVESTED))
    got = o.damage_item_upgrade("gliscor", CANON, {"tinglu": _defender()})
    assert got == "choiceband", got
