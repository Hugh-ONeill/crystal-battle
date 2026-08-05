"""Suspect #3 shadow probe (decision-rule shape): worst-case-across-worlds
vs visit-max, logged, never applied. The pure veto function is the whole
instrument — these pin its firing semantics."""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_player import _maximin_would_veto


def rr(*pairs):
    return [SimpleNamespace(move_choice=c, visits=v, total_score=s)
            for c, v, s in pairs]


def world(*pairs):
    return SimpleNamespace(side_one=rr(*pairs))


def test_fires_when_top_pick_collapses_in_one_world():
    # visit-max pick "boost" is great in world 0 (q=.8) but terrible in
    # world 1 (q=.2); "attack" holds .55 everywhere -> maximin vetoes
    ranked = rr(("boost", 900, 450.0), ("attack", 800, 440.0))
    results = [world(("boost", 500, 400.0), ("attack", 400, 220.0)),
               world(("boost", 400, 80.0), ("attack", 400, 220.0))]
    rec = _maximin_would_veto(ranked, results)
    assert rec["fired"] is True
    assert rec["worst_top"] == 0.2 and rec["worst_alt"] == 0.55


def test_quiet_when_top_pick_is_robust():
    ranked = rr(("attack", 900, 500.0), ("switch x", 700, 350.0))
    results = [world(("attack", 500, 300.0), ("switch x", 350, 175.0)),
               world(("attack", 400, 220.0), ("switch x", 350, 175.0))]
    rec = _maximin_would_veto(ranked, results)
    assert rec["fired"] is False


def test_incomparable_positions_return_none():
    ranked = rr(("attack", 900, 500.0), ("switch x", 700, 350.0))
    # "switch x" unexplored in world 1: a missing world Q makes min() lie
    results = [world(("attack", 500, 300.0), ("switch x", 350, 175.0)),
               world(("attack", 400, 220.0))]
    assert _maximin_would_veto(ranked, results) is None
    assert _maximin_would_veto(ranked, results[:1]) is None
    assert _maximin_would_veto(rr(("attack", 900, 500.0)), results) is None
