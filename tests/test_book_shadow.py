"""Opening-book shadow instrumentation: identification, conditions, margins.

Shadow mode must be inert — it records what the book WOULD have played and
returns nothing to the caller. These tests pin that, plus the two things that
would silently make the instrument useless: identifying the wrong team (the
bench hands us a per-lane file, so the paste name is gone) and evaluating a
condition it cannot actually check.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.opening_book import OpeningBook, _norm


class _Ranked:
    def __init__(self, move_choice, visits):
        self.move_choice = move_choice
        self.visits = visits


def _battle(active="torkoal", turn=1, weather=None, fields=(), screens=(),
            hp=1.0):
    class _E:            # poke-env enums are hashable; SimpleNamespace is not
        def __init__(self, name): self.name = name
        def __hash__(self): return hash(self.name)
        def __eq__(self, o): return getattr(o, "name", None) == self.name

    def enum(name):
        return _E(name)
    return SimpleNamespace(
        turn=turn,
        active_pokemon=SimpleNamespace(species=active,
                                       current_hp_fraction=hp),
        weather={enum(weather): 0} if weather else {},
        fields={enum(f): 0 for f in fields},
        side_conditions={enum(s): 0 for s in screens},
    )


def _sun():
    b = OpeningBook()
    b.identify(None, sorted(b._roster_index["sun1_torkoal"]))
    return b


def test_identifies_team_by_roster_not_by_filename():
    """The bench swaps a per-lane team file, so the original name is gone."""
    b = OpeningBook()
    key = b.identify("showdown/bench/webs_L1.team",
                     sorted(b._roster_index["sun1_torkoal"]))
    assert key == "sun1_torkoal"


def test_unknown_roster_books_nothing():
    b = OpeningBook()
    assert b.identify(None, ["Pikachu", "Magikarp"]) is None
    assert b.suggest(_battle()) is None
    assert b.lead() is None


def test_lead_is_recorded_and_compared():
    b = _sun()
    assert b.observe_lead("Torkoal")["agree"] is True
    assert b.observe_lead("Walking Wake")["agree"] is False


def test_first_matching_step_wins():
    b = _sun()
    step = b.suggest(_battle(active="torkoal", turn=1))
    assert step["do"] == "stealthrock"


def test_unused_gate_retires_a_step():
    """`unused` is what stops the book re-suggesting a move already played."""
    b = _sun()
    assert b.suggest(_battle(active="torkoal", turn=1))["do"] == "stealthrock"
    b.note_used("stealthrock")
    nxt = b.suggest(_battle(active="torkoal", turn=2, weather="SUNNYDAY"))
    assert nxt is None or nxt["do"] != "stealthrock"


def test_field_up_condition_reads_weather():
    b = _sun()
    b.note_used("stealthrock")
    assert b.suggest(_battle(active="torkoal", turn=2)) is None
    got = b.suggest(_battle(active="torkoal", turn=2, weather="SUNNYDAY"))
    assert got is not None and got["do"] == "rapidspin"


def test_wrong_active_does_not_fire():
    b = _sun()
    assert b.suggest(_battle(active="kingambit", turn=1)) is None


def test_observe_records_margin_and_is_inert():
    b = _sun()
    ranked = [_Ranked("protect", 700), _Ranked("stealthrock", 300)]
    before = [r.move_choice for r in ranked]
    entry = b.observe(_battle(active="torkoal", turn=1), ranked)
    assert entry["agree"] is False
    assert entry["book"] == "stealthrock" and entry["mcts"] == "protect"
    assert entry["book_rank"] == 1
    assert entry["margin"] == pytest.approx(0.4)
    assert [r.move_choice for r in ranked] == before, "shadow mutated ranked"


def test_agreement_is_reported_with_zero_margin():
    b = _sun()
    entry = b.observe(_battle(active="torkoal", turn=1),
                      [_Ranked("stealthrock", 900), _Ranked("protect", 100)])
    assert entry["agree"] is True and entry["margin"] == pytest.approx(0.0)


def test_book_move_absent_from_search_is_flagged_illegal():
    """A step naming a move the search never generated is the quietest way
    for the book to be useless, so it is recorded rather than dropped."""
    b = _sun()
    entry = b.observe(_battle(active="torkoal", turn=1),
                      [_Ranked("protect", 500)])
    assert entry["legal"] is False and entry["book_rank"] is None


def test_reset_clears_used_moves_between_games():
    b = _sun()
    b.note_used("stealthrock")
    b.reset()
    b.identify(None, sorted(b._roster_index["sun1_torkoal"]))
    assert b.suggest(_battle(active="torkoal", turn=1))["do"] == "stealthrock"
