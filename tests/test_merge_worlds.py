"""Merging MCTS results across sampled opponent worlds.

The old merge summed RAW visits across worlds. That silently weighted worlds
by how fast they happened to simulate: a world whose position resolves quickly
(fewer legal options, cheaper rollouts, an early terminal) accumulates more
iterations in the same milliseconds and therefore got more say in the vote,
for no principled reason. Found by reading foul-play's merge, which normalizes
each world's contribution by that world's own total visits.

We adopt the normalization. We deliberately do NOT adopt their second half —
weighting each world by its sampled probability — because our worlds are not
draws from a normalized belief distribution: world 0 is the deterministic
PS-curated read and the last world is deliberately SPEED-PESSIMISTIC, an
adversarial hedge against a Scarf. Down-weighting that hedge by its likelihood
is precisely what would defeat its purpose.
"""

from types import SimpleNamespace

from showdown.gen9_player import _merge_mcts_results


def world(*pairs):
    """world(("move", visits, total_score), ...) -> a result-shaped object."""
    return SimpleNamespace(side_one=[
        SimpleNamespace(move_choice=m, visits=v, total_score=s)
        for m, v, s in pairs
    ])


def ranking(merged):
    return [m.move_choice for m in merged]


def test_single_world_is_identity():
    """K=1 is the common case; merging must not perturb it."""
    w = world(("earthquake", 1000, 600.0), ("switch corviknight", 400, 180.0))
    merged = _merge_mcts_results([w])
    assert ranking(merged) == ["earthquake", "switch corviknight"]
    eq = merged[0]
    assert abs(eq.visits - 1000) < 1e-6
    assert abs(eq.total_score / eq.visits - 0.6) < 1e-9


def test_fast_world_does_not_dominate_the_vote():
    """Two worlds disagree. One simulated 10x more iterations purely because
    its position was cheaper. Under raw-visit summing its preference wins
    outright; normalized, the disagreement is a near-tie."""
    cheap = world(("icebeam", 900_000, 540_000.0), ("uturn", 100_000, 40_000.0))
    costly = world(("icebeam", 10_000, 4_000.0), ("uturn", 90_000, 54_000.0))
    merged = _merge_mcts_results([cheap, costly])
    by = {m.move_choice: m.visits for m in merged}
    # icebeam is 90% of the cheap world, uturn is 90% of the costly one:
    # normalized they must come out essentially equal
    assert abs(by["icebeam"] - by["uturn"]) / max(by.values()) < 0.05


def test_move_good_in_every_world_beats_a_one_world_spike():
    """The whole point of multiple worlds: consistency should win."""
    w1 = world(("knockoff", 500, 300.0), ("spikes", 900, 500.0))
    w2 = world(("knockoff", 500, 300.0), ("spikes", 10, 2.0))
    w3 = world(("knockoff", 500, 300.0), ("spikes", 10, 2.0))
    assert ranking(_merge_mcts_results([w1, w2, w3]))[0] == "knockoff"


def test_average_score_stays_meaningful():
    """_log_choice and the desk-read ledger both read total_score/visits."""
    w1 = world(("roost", 1000, 700.0))
    w2 = world(("roost", 1000, 300.0))
    merged = _merge_mcts_results([w1, w2])
    roost = merged[0]
    assert abs(roost.total_score / roost.visits - 0.5) < 1e-6


def test_move_absent_from_one_world_is_penalized_not_dropped():
    """A move only legal/searched in some worlds still competes, but the
    worlds that never searched it count as zero support."""
    w1 = world(("tera blast", 800, 500.0), ("protect", 200, 90.0))
    w2 = world(("protect", 1000, 450.0))
    merged = _merge_mcts_results([w1, w2])
    names = ranking(merged)
    assert "tera blast" in names and "protect" in names
    assert names[0] == "protect"


def test_zero_visit_world_is_ignored_not_fatal():
    live = world(("earthquake", 500, 300.0))
    dead = SimpleNamespace(side_one=[])
    merged = _merge_mcts_results([live, dead])
    assert ranking(merged) == ["earthquake"]


def test_no_results_is_empty():
    assert _merge_mcts_results([]) == []
