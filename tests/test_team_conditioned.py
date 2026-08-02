"""Partial-core backoff for the replay archetype tier (CB_TEAM_COND_OVERLAP).

The exact-6-roster `team_match` fires on 70.5% of the rosters we actually
face; >=5 shared species reaches 87.1% and >=4 reaches 98.4% (measured over
730 booked games, 2026-08-02). Team-conditioning is worth extending because a
mon's build genuinely shifts with its neighbours — median total-variation
from the pooled move distribution 0.23-0.39 vs a resample-null of 0.09-0.12.

DEFAULT OFF: this is a world-composition change, and those are 0-for-3 on
winrate, so it ships gated and awaits its own paired A/B.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.replay_sets import ReplaySetsIndex


@pytest.fixture(scope="module")
def idx():
    return ReplaySetsIndex()


def a_real_core(idx, min_count=5):
    for key, team in idx.teams.items():
        if team.get("count", 0) >= min_count:
            return key.split("|"), team
    pytest.skip("no core in the pool")


def test_partial_match_finds_what_exact_match_misses(idx):
    core, _ = a_real_core(idx)
    swapped = core[:5] + ["pikachu"]          # 5 of 6 shared, never an exact core
    assert idx.team_match(swapped) is None
    m = idx.partial_team_match(swapped, min_overlap=5)
    assert m and m["partial"] and m["count"] > 0
    assert m["mons"], "merged entry must carry per-species data"


def test_lower_overlap_is_a_superset_of_higher(idx):
    core, _ = a_real_core(idx)
    probe = core[:4] + ["pikachu", "ditto"]
    m5 = idx.partial_team_match(probe, min_overlap=5)
    m4 = idx.partial_team_match(probe, min_overlap=4)
    assert m4 is not None
    if m5 is not None:
        assert m4["count"] >= m5["count"]


def test_merged_entry_has_the_shape_pick_moves_expects(idx):
    core, _ = a_real_core(idx)
    m = idx.partial_team_match(core, min_overlap=5)
    assert m is not None
    sp = next(iter(m["mons"]))
    picked = idx.pick_moves(sp, team=m)
    assert picked is None or isinstance(picked, list)


def test_min_count_gate_excludes_home_brews(idx):
    core, _ = a_real_core(idx)
    high = idx.partial_team_match(core, min_overlap=4, min_count=1000000)
    assert high is None          # nothing clears an absurd gate


def test_result_is_memoised(idx):
    core, _ = a_real_core(idx)
    first = idx.partial_team_match(core, min_overlap=4)
    second = idx.partial_team_match(core, min_overlap=4)
    assert first is second       # same object: the scan ran once


def test_default_is_off(monkeypatch):
    """Absent the env flag the translator must not use the backoff at all."""
    monkeypatch.delenv("CB_TEAM_COND_OVERLAP", raising=False)
    import importlib
    import showdown.gen9_translator as t
    importlib.reload(t)
    assert t._TEAM_COND_OVERLAP == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
