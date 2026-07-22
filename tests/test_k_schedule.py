"""The decaying world schedule: many sampled worlds while the opponent's sets
are unknown, collapsing to one as they get revealed.

Rationale, all measured: set uncertainty is highest early while the value of
DEPTH is highest late, so breadth is worth most exactly where depth is worth
least. Per-world depth holds at 96% for K=4 and 91% for K=8 at a 300ms budget
before thread contention bites (69% at K=16, 53% at K=24), and steady-state
translation is ~1.7ms/world, so extra early worlds are close to free.

The memory cap is the load-bearing part: one world's tree is ~63MB at 300ms
but ~592MB at the 6000ms grind cap, and K worlds hold K trees. A constant K=8
would collide with the grind budget in exactly the late-game positions where
the budget is largest.
"""

from types import SimpleNamespace

from showdown.gen9_player import Gen9PokeEnginePlayer as P


def player(**kw):
    """A real player instance WITHOUT running __init__ (which would open a
    websocket). The schedule methods only touch the attributes set here."""
    obj = object.__new__(P)
    cfg = dict(_k_schedule=True, _k_max=8, _k_ms_product=2400,
               _collapse_moves=14, _collapse_mons=5, _verbose=False,
               _set_samples=2, _collapse_turn=25)
    cfg.update(kw)
    for k, v in cfg.items():
        setattr(obj, k, v)
    obj._airi_engine_beat = lambda *a, **k: None   # collapse announce path
    return obj


def battle(revealed=0, acted=0, turn=1):
    mons = []
    for i in range(6):
        mons.append(SimpleNamespace(
            moves={f"m{j}": None for j in range(
                (revealed // max(1, acted)) if i < acted and acted else 0)}
            if i < acted else {}))
    # distribute an exact revealed-move count across the acted mons
    if acted:
        total = sum(len(m.moves) for m in mons)
        i = 0
        while total < revealed and i < acted:
            mons[i].moves[f"x{total}"] = None
            total += 1
            i = (i + 1) % acted
    return SimpleNamespace(opponent_team={str(i): m for i, m in enumerate(mons)},
                           turn=turn)


class TestCoverage:
    def test_zero_when_nothing_revealed(self):
        assert player()._opp_coverage(battle()) == 0.0

    def test_full_when_move_threshold_met(self):
        assert player()._opp_coverage(battle(revealed=14, acted=4)) == 1.0

    def test_narrow_movepool_still_reaches_full_via_mons_acted(self):
        """The stall exploit probe revealed only 12 moves across a fully-played
        team; a moves-only signal would never collapse against exactly the
        archetype the grind package targets."""
        cov = player()._opp_coverage(battle(revealed=10, acted=5))
        assert cov == 1.0


class TestSchedule:
    def test_starts_at_k_max_when_nothing_is_known(self):
        assert player()._scheduled_samples(battle(), 300) == 8

    def test_collapses_to_one_when_fully_revealed(self):
        assert P._scheduled_samples(
            player(), battle(revealed=14, acted=4), 300) == 1

    def test_decays_monotonically_with_coverage(self):
        p = player()
        ks = [p._scheduled_samples(battle(revealed=r, acted=1), 300)
              for r in (0, 4, 8, 12, 14)]
        assert ks[0] == 8 and ks[-1] == 1
        assert all(a >= b for a, b in zip(ks, ks[1:])), ks

    def test_memory_cap_binds_at_the_grind_budget(self):
        """K x budget is what allocates trees. At the 6s grind cap the default
        product must force a single world even at zero coverage."""
        assert player()._scheduled_samples(battle(), 6000) == 1

    def test_memory_cap_allows_full_breadth_at_bench_budget(self):
        assert player()._scheduled_samples(battle(), 300) == 8

    def test_never_below_one(self):
        p = player(_k_ms_product=1)
        assert p._scheduled_samples(battle(revealed=14, acted=6), 99999) == 1

    def test_k_max_of_one_is_a_constant(self):
        p = player(_k_max=1)
        assert p._scheduled_samples(battle(), 300) == 1


class TestGateIntegration:
    def test_schedule_off_preserves_the_old_constant(self):
        p = player(_k_schedule=False)
        assert p._effective_samples(battle(turn=1), 300) == 2

    def test_schedule_off_still_collapses_late(self):
        p = player(_k_schedule=False)
        assert P._effective_samples(
            p, battle(revealed=14, acted=4, turn=30), 300) == 1

    def test_hard_collapse_remains_a_floor_when_scheduled(self):
        """Even if the schedule wanted more, the old gate pins it to 1."""
        p = player(_k_ms_product=999999)
        assert P._effective_samples(
            p, battle(revealed=14, acted=4, turn=30), 300) == 1

    def test_schedule_applies_before_the_collapse_turn(self):
        p = player()
        assert p._effective_samples(battle(turn=1), 300) == 8

    def test_missing_budget_falls_back_to_the_constant(self):
        p = player()
        assert p._effective_samples(battle(turn=1), None) == 2
