"""The book must not blend the same species across different teams.

Every opponent reuses species across rosters — richwoman covers 17 of her 34
species over 12 rosters, and 100% of games involve at least one — and the
build genuinely differs: her Ting-Lu runs Protect on one team, Spikes on
another and neither on a third, and her Dragonite is a Dragon Dance sweeper
on one roster beside a mixed attacker on the next. A species-keyed blend
yields a set no team of hers actually runs.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_translator import Gen9Translator

BOOK = {
    "sets": {  # the blend: a Frankenstein of both builds
        "Ting-Lu": {"moves": {"stealthrock": 40, "earthquake": 30,
                              "protect": 20, "spikes": 18, "ruination": 15},
                    "items": {}, "abilities": {}, "tera": {}},
    },
    "sets_by_roster": {
        "alomomola|ironvaliant|kingambit|latios|pecharunt|tinglu": {
            "Ting-Lu": {"moves": {"stealthrock": 24, "protect": 23,
                                  "earthquake": 22, "whirlwind": 17},
                        "items": {}, "abilities": {}, "tera": {}},
        },
        "darkrai|gholdengo|gliscor|primarina|tinglu|zapdos": {
            "Ting-Lu": {"moves": {"ruination": 25, "stealthrock": 21,
                                  "whirlwind": 17, "spikes": 10},
                        "items": {}, "abilities": {}, "tera": {}},
        },
    },
}


def tr_with(roster):
    t = Gen9Translator(set_source="gen9ou")
    t._book = BOOK
    t._book_min_obs = 3
    t._opp_roster = roster
    return t


def test_the_matching_roster_wins():
    t = tr_with("alomomola|ironvaliant|kingambit|latios|pecharunt|tinglu")
    s = t._book_set("tinglu")
    assert s is not None
    assert "protect" in s["moves"]
    assert "spikes" not in s["moves"], "took moves from the other roster"


def test_a_different_roster_gives_the_other_build():
    t = tr_with("darkrai|gholdengo|gliscor|primarina|tinglu|zapdos")
    s = t._book_set("tinglu")
    assert "spikes" in s["moves"] or "ruination" in s["moves"]
    assert "protect" not in s["moves"], "took moves from the other roster"


def test_an_unseen_roster_falls_back_to_the_blend():
    t = tr_with("aaa|bbb|ccc|ddd|eee|fff")
    s = t._book_set("tinglu")
    assert s is not None, "must still answer from the species-level blend"


def test_no_roster_context_still_works():
    t = Gen9Translator(set_source="gen9ou")
    t._book, t._book_min_obs = BOOK, 3
    assert t._book_set("tinglu") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
