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


CHOICE = {"choiceband", "choicespecs", "choicescarf"}


@pytest.fixture(scope="module")
def chaos_shares():
    from showdown.chaos_stats import ChaosStats
    cs = ChaosStats(format="gen9ou")
    shares = {}
    for st in cs.pokemon.values():
        for i, p in st._items.items():
            shares[i] = shares.get(i, 0.0) + p * st.usage
    return cs, shares


def test_invisible_items_are_never_treated_as_evidence(idx, chaos_shares):
    """Choice items are 14.06% of real usage and 0.09% of replay
    observations because they emit no protocol event. Boots, Assault Vest,
    Light Clay and the masks are equally invisible. If any of these entered
    the observable set, conditioning would delete them from our beliefs."""
    _, shares = chaos_shares
    obs = idx.observable_items(shares)
    assert not (obs & CHOICE), "choice items must never be observable"
    for invisible in ("heavydutyboots", "assaultvest", "lightclay",
                      "loadeddice", "wellspringmask"):
        assert invisible not in obs, invisible
    assert "leftovers" in obs and "rockyhelmet" in obs


def test_choice_probability_survives_conditioning(idx, chaos_shares):
    """The whole safety property: replay evidence re-allocates only the mass
    chaos gives to VISIBLE items, so P(some choice item) is preserved."""
    import random
    cs, shares = chaos_shares
    core = next(k for k, v in idx.teams.items()
                if v.get("count", 0) >= 20).split("|")
    team = idx.team_match(core)
    rng = random.Random(3)
    tested = 0
    for sp in core:
        st = cs.pokemon.get(sp)
        if st is None:
            continue
        probs = dict(st._items)
        tot = sum(probs.values()) or 1
        prior = sum(p for i, p in probs.items() if i in CHOICE) / tot
        draws = [idx.conditioned_item(sp, probs, team, shares, rng)
                 for _ in range(400)]
        draws = [d for d in draws if d]
        if len(draws) < 100 or prior < 0.05:
            continue
        tested += 1
        got = sum(1 for d in draws if d in CHOICE) / len(draws)
        assert abs(got - prior) < 0.06, f"{sp}: {prior:.2%} -> {got:.2%}"
    assert tested, "no species with meaningful choice mass exercised"


def test_thin_evidence_falls_through(idx, chaos_shares):
    import random
    cs, shares = chaos_shares
    core = next(k for k, v in idx.teams.items()
                if v.get("count", 0) >= 20).split("|")
    team = idx.team_match(core)
    sp = core[0]
    st = cs.pokemon.get(sp)
    if st is None:
        pytest.skip("no chaos entry")
    out = idx.conditioned_item(sp, dict(st._items), team, shares,
                               random.Random(1), min_obs=10 ** 9)
    assert out is None


def test_no_team_means_no_conditioning(idx, chaos_shares):
    import random
    cs, shares = chaos_shares
    st = next(iter(cs.pokemon.values()))
    assert idx.conditioned_item("kingambit", dict(st._items), None, shares,
                                random.Random(1)) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
