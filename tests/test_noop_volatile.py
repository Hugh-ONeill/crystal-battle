"""Volatile-duplicate no-ops: re-applying what the target already has fails.

User-raised 2026-08-02: Taunting an already-taunted mon is the type case.
It is the same flat-eval trap as the maxed-hazard repeat — the state after
the failed move is identical to the state after doing nothing, so a search
whose eval cannot separate them will spend the turn on it (measured on
hazards as 19 consecutive Spikes, 16 wasted, at 5s).
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    """These filters are GATED OFF by default (CB_NOOP_FAIL) — a failing move
    is a zero-damage idle, which is the right play against Mirror Coat or
    Counter, so removing it at the root can force a worse attack. Tests
    exercise the enabled behaviour explicitly."""
    import showdown.gen9_player as gp
    monkeypatch.setattr(gp, "_NOOP_FAIL_ON", True)

from poke_env.battle.effect import Effect
from showdown.gen9_player import _is_noop_volatile


@dataclass(frozen=True)
class Cond:
    name: str


def battle(foe_effects=(), self_effects=(), side=(), weather=()):
    return NS(
        opponent_active_pokemon=NS(effects={e: 1 for e in foe_effects}),
        active_pokemon=NS(effects={e: 1 for e in self_effects}),
        side_conditions={Cond(s): 1 for s in side},
        weather={Cond(w): 1 for w in weather},
    )


def test_taunt_into_an_already_taunted_target_is_a_noop():
    assert _is_noop_volatile("taunt", battle(foe_effects=[Effect.TAUNT]))


def test_taunt_into_an_untaunted_target_is_fine():
    assert not _is_noop_volatile("taunt", battle())


@pytest.mark.parametrize("move,effect", [
    ("encore", Effect.ENCORE),
    ("leechseed", Effect.LEECH_SEED),
    ("yawn", Effect.YAWN),
    ("disable", Effect.DISABLE),
    ("attract", Effect.ATTRACT),
    ("confuseray", Effect.CONFUSION),
])
def test_other_foe_volatiles(move, effect):
    assert _is_noop_volatile(move, battle(foe_effects=[effect]))
    assert not _is_noop_volatile(move, battle())


def test_substitute_is_checked_on_our_own_side():
    assert _is_noop_volatile("substitute",
                             battle(self_effects=[Effect.SUBSTITUTE]))
    # the OPPONENT having a sub says nothing about our ability to make one
    assert not _is_noop_volatile("substitute",
                                 battle(foe_effects=[Effect.SUBSTITUTE]))


@pytest.mark.parametrize("move,cond", [
    ("reflect", "REFLECT"),
    ("lightscreen", "LIGHT_SCREEN"),
    ("tailwind", "TAILWIND"),
])
def test_our_side_conditions_already_standing(move, cond):
    assert _is_noop_volatile(move, battle(side=[cond]))
    assert not _is_noop_volatile(move, battle())


def test_aurora_veil_requires_snow():
    """It fails outright without snow, whether or not a veil is up."""
    assert _is_noop_volatile("auroraveil", battle())
    assert not _is_noop_volatile("auroraveil", battle(weather=["SNOW"]))
    assert _is_noop_volatile("auroraveil",
                             battle(side=["AURORA_VEIL"], weather=["SNOW"]))


def test_trick_room_is_deliberately_not_filtered():
    """Re-using Trick Room ENDS it early — a real choice, not a no-op."""
    assert not _is_noop_volatile("trickroom", battle(side=["TRICK_ROOM"]))


def pending_battle(hp=1.0, self_effects=(), foe_effects=(), side=()):
    return NS(
        active_pokemon=NS(effects={e: 1 for e in self_effects},
                          current_hp_fraction=hp),
        opponent_active_pokemon=NS(effects={e: 1 for e in foe_effects},
                                   current_hp_fraction=1.0),
        side_conditions={Cond(s): 1 for s in side},
        weather={},
    )


def test_future_sight_while_one_is_pending():
    from showdown.gen9_player import _is_noop_pending
    assert _is_noop_pending(
        "futuresight", pending_battle(foe_effects=[Effect.FUTURE_SIGHT]))
    assert not _is_noop_pending("futuresight", pending_battle())


def test_wish_while_one_is_pending():
    from showdown.gen9_player import _is_noop_pending
    assert _is_noop_pending("wish", pending_battle(side=["WISH"]))
    assert not _is_noop_pending("wish", pending_battle())


def test_substitute_below_its_own_cost():
    from showdown.gen9_player import _is_noop_pending
    assert _is_noop_pending("substitute", pending_battle(hp=0.20))
    assert not _is_noop_pending("substitute", pending_battle(hp=0.80))


def test_recovery_at_full_hp():
    from showdown.gen9_player import _is_noop_pending
    for mv in ("roost", "recover", "softboiled", "synthesis"):
        assert _is_noop_pending(mv, pending_battle(hp=1.0)), mv
        assert not _is_noop_pending(mv, pending_battle(hp=0.9)), mv


def test_prediction_gambles_are_never_filtered():
    """Sucker Punch and Thunderclap fail by DESIGN when the read is wrong —
    89 and 42 fails in our logs, all of them correct attempts. Filtering
    them would remove the move's whole purpose."""
    from showdown.gen9_player import _is_noop_pending, _is_noop_volatile
    for mv in ("suckerpunch", "thunderclap"):
        assert not _is_noop_pending(mv, pending_battle())
        assert not _is_noop_volatile(mv, battle())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_filters_are_off_by_default(monkeypatch):
    """Default behaviour must be unchanged: the Mirror Coat specimen showed a
    guaranteed-fail move is sometimes the BEST play, so this ships gated."""
    import importlib
    monkeypatch.delenv("CB_NOOP_FAIL", raising=False)
    import showdown.gen9_player as gp
    importlib.reload(gp)
    assert gp._NOOP_FAIL_ON is False
    assert not gp._is_noop_volatile("taunt", battle(foe_effects=[Effect.TAUNT]))
