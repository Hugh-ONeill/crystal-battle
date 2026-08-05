"""Negative-evidence Heavy-Duty Boots (gc-0020): a switch-in over Stealth
Rock that takes zero chip is Boots — unless Magic Guard can explain it.
Drives the REAL BattleObservations scanner over crafted protocol."""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import BattleObservations


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


SR = "|-sidestart|p2: Opp|move: Stealth Rock"
SPK = "|-sidestart|p2: Opp|move: Spikes"
GRAV = "|-fieldstart|move: Gravity"


def test_zero_chip_over_rocks_infers_boots():
    o = _obs([SR,
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|turn|5"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_taking_sr_chip_cancels_boots():
    o = _obs([SR,
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|-damage|p2a: Corviknight|94/100|[from] Stealth Rock",
              "|turn|5"])
    assert o.boots_inferred("corviknight") is None
    assert "corviknight" not in o.boots


def test_magic_guard_species_is_ambiguous_not_promoted():
    o = _obs([SR,
              "|switch|p2a: Clefable|Clefable, F|100/100",
              "|turn|5"])
    assert o.boots_inferred("clefable") is None       # search must not guess
    assert "clefable" in o.boots_ambiguous            # but recorded for hedge


def test_no_rocks_no_evidence():
    o = _obs(["|switch|p2a: Corviknight|Corviknight, M|100/100", "|turn|5"])
    assert o.boots_inferred("corviknight") is None


def test_mon_already_in_when_rocks_set_is_not_evidence():
    # Corviknight was active BEFORE we set rocks; it never switched over them
    o = _obs(["|switch|p2a: Corviknight|Corviknight, M|100/100",
              SR,
              "|turn|5"])
    assert o.boots_inferred("corviknight") is None


def test_window_closes_on_next_move_not_just_turn():
    o = _obs([SR,
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|move|p2a: Corviknight|Roost|p2a: Corviknight"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_final_switch_resolves_at_end_of_batch():
    # replay ends right after the switch (no following turn/move)
    o = _obs([SR, "|switch|p2a: Corviknight|Corviknight, M|100/100"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_court_change_moves_rocks_to_the_other_side():
    # rocks start on OUR side (p1); Court Change flips them onto p2's side,
    # then their mon switches over them
    o = _obs(["|-sidestart|p1: Us|move: Stealth Rock",
              "|-swapsideconditions|",
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|turn|5"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


# --- Spikes: grounded-only evidence -----------------------------------------

def test_grounded_mon_avoiding_spikes_infers_boots():
    # Great Tusk (Ground/Fighting, no Levitate) is grounded -> avoiding
    # Spikes chip is Boots evidence just like Stealth Rock
    o = _obs([SPK, "|switch|p2a: Great Tusk|Great Tusk|100/100", "|turn|5"])
    assert o.boots_inferred("greattusk") == "heavydutyboots"


def test_flying_mon_avoiding_spikes_is_no_evidence():
    # Corviknight is Flying -> it dodges Spikes legitimately, no signal
    o = _obs([SPK, "|switch|p2a: Corviknight|Corviknight, M|100/100", "|turn|5"])
    assert o.boots_inferred("corviknight") is None


def test_possible_levitate_avoiding_spikes_is_no_evidence():
    # Bronzong can run Levitate -> can't prove it's grounded
    o = _obs([SPK, "|switch|p2a: Bronzong|Bronzong|100/100", "|turn|5"])
    assert o.boots_inferred("bronzong") is None


def test_air_balloon_voids_spikes_evidence():
    o = _obs([SPK, "|switch|p2a: Great Tusk|Great Tusk|100/100",
              "|-item|p2a: Great Tusk|Air Balloon", "|turn|5"])
    assert o.boots_inferred("greattusk") is None


def test_spikes_chip_cancels_boots():
    o = _obs([SPK, "|switch|p2a: Great Tusk|Great Tusk|100/100",
              "|-damage|p2a: Great Tusk|88/100|[from] Spikes", "|turn|5"])
    assert o.boots_inferred("greattusk") is None


def test_stealth_rock_still_proves_flying_mon():
    # SR hits everything, so a Flying mon avoiding SR+Spikes is still Boots
    o = _obs([SR, SPK, "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|turn|5"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


# --- Gravity: grounds everyone ----------------------------------------------

def test_gravity_grounds_flying_for_spikes_evidence():
    o = _obs([GRAV, SPK, "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|turn|5"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_gravity_overrides_air_balloon():
    o = _obs([GRAV, SPK, "|switch|p2a: Great Tusk|Great Tusk|100/100",
              "|-item|p2a: Great Tusk|Air Balloon", "|turn|5"])
    assert o.boots_inferred("greattusk") == "heavydutyboots"


def test_gravity_ended_reverts_to_normal():
    o = _obs([GRAV, "|-fieldend|move: Gravity", SPK,
              "|switch|p2a: Corviknight|Corviknight, M|100/100", "|turn|5"])
    assert o.boots_inferred("corviknight") is None


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")


# ---- positive proofs from the non-damage hazards (2026-08-05) -----------

TSPIKES = "|-sidestart|p2: Opp|move: Toxic Spikes"


def test_entry_poison_over_tspikes_proves_no_boots():
    o = _obs([TSPIKES,
              "|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|-status|p2a: Garchomp|psn",
              "|turn|5"])
    assert "heavydutyboots" in o.forbidden("garchomp")
    o2 = _obs([TSPIKES, TSPIKES,     # two layers -> tox
               "|switch|p2a: Garchomp|Garchomp, M|100/100",
               "|-status|p2a: Garchomp|tox",
               "|turn|5"])
    assert "heavydutyboots" in o2.forbidden("garchomp")


def test_late_poison_is_not_entry_evidence():
    # poisoned by a MOVE after entering: the window closed at the move line
    o = _obs([TSPIKES,
              "|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Earthquake|p1a: X",
              "|-status|p2a: Garchomp|psn",
              "|turn|5"])
    assert "heavydutyboots" not in o.forbidden("garchomp")


def test_no_tspikes_no_window():
    o = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|-status|p2a: Garchomp|psn",
              "|turn|5"])
    assert "heavydutyboots" not in o.forbidden("garchomp")


def test_web_activation_proves_no_boots_and_cancels_the_latch():
    o = _obs([SR, "|-sidestart|p2: Opp|move: Sticky Web",
              "|switch|p2a: Clefable|Clefable, F|100/100",
              "|-activate|p2a: Clefable|move: Sticky Web",
              "|-unboost|p2a: Clefable|spe|1",
              "|turn|5"])
    assert "heavydutyboots" in o.forbidden("clefable")
    # web fired but SR didn't chip: the zero-chip latch must NOT conclude
    # boots (bootlessness is proven; the silent SR points at Magic Guard)
    assert "clefable" not in o.boots
    assert "clefable" not in o.boots_ambiguous


def test_web_plus_silent_rocks_proves_magic_guard():
    """Web-loud + rocks-silent is a DEDUCTION: web's -activate is
    sim-guaranteed grounded+bootless, and only Magic Guard blocks SR chip
    in gen9. The Clefable coinflip (MG vs Unaware), resolved on entry."""
    o = _obs([SR, "|-sidestart|p2: Opp|move: Sticky Web",
              "|switch|p2a: Clefable|Clefable, F|100/100",
              "|-activate|p2a: Clefable|move: Sticky Web",
              "|-unboost|p2a: Clefable|spe|1",
              "|turn|5"])
    assert "clefable" in o.magic_guard


def test_no_rocks_means_web_proves_only_bootlessness():
    o = _obs(["|-sidestart|p2: Opp|move: Sticky Web",
              "|switch|p2a: Clefable|Clefable, F|100/100",
              "|-activate|p2a: Clefable|move: Sticky Web",
              "|turn|5"])
    assert "clefable" not in o.magic_guard
    assert "heavydutyboots" in o.forbidden("clefable")


def test_mg_incapable_species_never_asserted():
    o = _obs([SR, "|-sidestart|p2: Opp|move: Sticky Web",
              "|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|-activate|p2a: Garchomp|move: Sticky Web",
              "|turn|5"])
    assert "garchomp" not in o.magic_guard


def test_sr_chip_disproves_magic_guard():
    o = _obs([SR,
              "|switch|p2a: Clefable|Clefable, F|100/100",
              "|-damage|p2a: Clefable|94/100|[from] Stealth Rock",
              "|turn|5"])
    assert "magicguard" in o.ability_forbidden("clefable")


def test_translator_adopts_deduced_magic_guard():
    import logging
    from poke_env.battle.battle import Battle
    from showdown.gen9_translator import Gen9Translator
    from tests.test_gen9_translator import REQUEST
    b = Battle("battle-gen9monotype-test-7", "wizbot",
               logging.getLogger("t"), gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F",
                     "323/323"])
    b.parse_message(["", "-sidestart", "p2: opp", "move: Stealth Rock"])
    b.parse_message(["", "-sidestart", "p2: opp", "move: Sticky Web"])
    b.parse_message(["", "switch", "p2a: Clefable", "Clefable, F",
                     "100/100"])
    b.parse_message(["", "-activate", "p2a: Clefable", "move: Sticky Web"])
    b.parse_message(["", "turn", "2"])
    state = Gen9Translator().translate(b)
    clef = next(p for p in state.side_two.pokemon if p.id == "clefable")
    assert clef.ability == "magicguard"
