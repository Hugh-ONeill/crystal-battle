"""Endgame-solver tera gate.

_both_teras_spent must read tera-spent from the poke-env battle
(Pokemon.is_terastallized survives a faint), NOT the translated engine state
(where the translator rebuilds fainted mons as blank create_fainted() dummies
with terastallized=False). The old state-based gate locked the solver out of
every endgame where a tera'd mon had already fainted — i.e. almost all of
them — so the live solver effectively never fired. These pin the fix."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_player import Gen9PokeEnginePlayer

_spent = Gen9PokeEnginePlayer._both_teras_spent


def _mon(tera, fainted=False):
    return SimpleNamespace(is_terastallized=tera, fainted=fainted)


def _battle(ours, theirs):
    return SimpleNamespace(
        team={str(i): m for i, m in enumerate(ours)},
        opponent_team={str(i): m for i, m in enumerate(theirs)})


def test_both_sides_tera_alive():
    assert _spent(_battle([_mon(True), _mon(False)],
                          [_mon(True), _mon(False)]))


def test_tera_survives_faint():
    # the regression this fixes: both tera'd mons have FAINTED but still count
    assert _spent(_battle([_mon(True, fainted=True), _mon(False)],
                          [_mon(True, fainted=True), _mon(False)]))


def test_one_side_never_tera_blocks():
    assert not _spent(_battle([_mon(True)], [_mon(False), _mon(False)]))


def test_neither_side_tera_blocks():
    assert not _spent(_battle([_mon(False)], [_mon(False)]))


def test_missing_attr_is_safe():
    # a mon without is_terastallized must not raise (getattr default False)
    b = SimpleNamespace(team={"0": SimpleNamespace()},
                        opponent_team={"0": SimpleNamespace()})
    assert not _spent(b)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
