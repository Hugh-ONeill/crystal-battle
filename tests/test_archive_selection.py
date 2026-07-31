# Book-weighted archive selection + the unrevealed-only gate.
#
# The archive was shelved after three winrate gates. The 2026-07-30 belief
# measurement found why: it CONTAINS the opponent's team (95.3% move / 97.3%
# item / 90.6% tera for the best candidate vs richwoman) but a blind draw got
# 69.5/68.7/42.1 — a SELECTION problem. Ranking consistent candidates by
# scouting-book agreement lifts a top-1 pick to 82.9/91.5/74.1. These tests
# pin the two mechanisms that implements.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.team_archive import TeamArchive


def _mon(species, moves, item=None, tera=None, ability=None):
    return {"species": species, "moves": list(moves), "item": item,
            "tera_type": tera, "ability": ability}


def test_book_score_prefers_the_observed_variant():
    """A candidate matching what we have SEEN this opponent do outranks a
    plausible-but-different variant of the same species."""
    seen = _mon("tinglu", ["stealthrock", "whirlwind", "earthquake", "ruination"],
                item="leftovers", tera="fairy")
    unseen = _mon("tinglu", ["spikes", "bodypress", "rest", "sleeptalk"],
                  item="rockyhelmet", tera="ghost")
    book = {"Ting-Lu": {"moves": {"stealthrock": 65, "whirlwind": 55,
                                  "earthquake": 56, "ruination": 55},
                        "items": {"leftovers": 30}, "tera": {"fairy": 4}}}
    hi = TeamArchive._book_score([seen], book)
    lo = TeamArchive._book_score([unseen], book)
    assert hi > lo, f"observed variant must score higher: {hi} !> {lo}"
    assert hi > 0.9 and lo < 0.2


def test_book_score_weights_item_and_tera_not_just_moves():
    """Scoring on moves alone left item/tera flat in the measurement; each is
    weighted like a move slot so selection can act on them."""
    book = {"Zapdos": {"moves": {}, "items": {"heavydutyboots": 20},
                       "tera": {"steel": 5}}}
    right = _mon("zapdos", [], item="heavydutyboots", tera="steel")
    wrong = _mon("zapdos", [], item="leftovers", tera="water")
    assert TeamArchive._book_score([right], book) == 1.0
    assert TeamArchive._book_score([wrong], book) == 0.0


def test_book_score_is_neutral_without_history():
    """A cold-start opponent must not be scored into a false preference."""
    m = _mon("greattusk", ["headlongrush", "rapidspin"], item="boosterenergy")
    assert TeamArchive._book_score([m], {}) == 0.0
    assert TeamArchive._book_score([m], {"Other": {"moves": {"tackle": 1}}}) == 0.0


def test_archive_gate_answers_only_unrevealed_by_default(monkeypatch):
    """Default `unrevealed` mode: the archive answers for a mon that has shown
    nothing, and steps aside once a move is revealed so the observation-
    filtered tiers handle it — capping correlated-wrong exposure."""
    from showdown.gen9_translator import Gen9Translator
    t = Gen9Translator.__new__(Gen9Translator)
    monkeypatch.delenv("CB_ARCHIVE_MODE", raising=False)
    t._archive_mode = None
    assert t._archive_covers(()) is True
    t._archive_mode = None
    assert t._archive_covers(("stealthrock",)) is False


def test_archive_gate_all_mode_reproduces_v1(monkeypatch):
    """`all` restores the v1 whole-team override so the two can be A/B'd."""
    from showdown.gen9_translator import Gen9Translator
    t = Gen9Translator.__new__(Gen9Translator)
    monkeypatch.setenv("CB_ARCHIVE_MODE", "all")
    t._archive_mode = None
    assert t._archive_covers(()) is True
    assert t._archive_covers(("stealthrock",)) is True
