"""Set-inference belief-delta beats: the player diffs the search's
confirmed inferences and emits a one-time set_reveal per new belief.
Driven through the same director seam the live player uses, with a
stub translator/observations so no engine or battle is needed."""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_player import Gen9PokeEnginePlayer as P, _belief_prose
from showdown.beat_director import Director


def _player():
    """A Gen9PokeEnginePlayer skeleton with just the belief-delta wiring
    (no poke-env base __init__)."""
    p = P.__new__(P)
    p._airi = object()               # non-None so guards pass
    p._director = Director()
    p._announced_beliefs = {}
    p._translator = SimpleNamespace(_obs=SimpleNamespace(confirmed={}))
    return p


def _pending_belief_beats(director):
    return [b for b in
            (__import__("showdown.beat_director", fromlist=["classify"])
             .classify(ev) for ev in director._pending)
            if b and b.beat == "set_reveal"]


def test_belief_delta_fires_once_per_belief():
    p = _player()
    p._translator._obs.confirmed = {"ironvaliant": "choicescarf"}
    p._emit_belief_deltas()
    beats = _pending_belief_beats(p._director)
    assert len(beats) == 1 and "Choice Scarf" in beats[0].prose
    assert beats[0].persona == "either"
    # same belief again -> no repeat
    p._director._pending.clear()
    p._emit_belief_deltas()
    assert _pending_belief_beats(p._director) == []


def test_new_belief_on_same_mon_re_fires():
    p = _player()
    p._translator._obs.confirmed = {"kingambit": "lifeorb"}
    p._emit_belief_deltas()
    p._director._pending.clear()
    # damage evidence escalated the belief -> a new reveal
    p._translator._obs.confirmed = {"kingambit": "choiceband"}
    p._emit_belief_deltas()
    beats = _pending_belief_beats(p._director)
    assert len(beats) == 1 and "Choice Band" in beats[0].prose


def test_reset_clears_announced():
    p = _player()
    p._translator._obs.confirmed = {"dragapult": "choicescarf"}
    p._emit_belief_deltas()
    assert p._announced_beliefs                      # populated
    # new battle resets the announced set (mirrors _airi_new_battle)
    p._announced_beliefs = {}
    p._emit_belief_deltas()
    assert _pending_belief_beats(p._director)         # fires fresh again


def test_no_obs_is_safe():
    p = _player()
    p._translator = SimpleNamespace(_obs=None)
    p._emit_belief_deltas()                           # no raise
    assert p._director._pending == []


def test_belief_prose_labels():
    assert "Choice Scarf" in _belief_prose("Iron Valiant", "choicescarf")
    assert "Life Orb" in _belief_prose("Dragapult", "lifeorb")
    assert "Choice Band" in _belief_prose("Kingambit", "choiceband")
    # unknown id degrades to a generic boosting-item phrasing, still named
    assert "boosting item" in _belief_prose("X", "weirditem")


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
