"""The opening book must name moves the teams actually carry.

A step naming a move nobody has is a silent no-op — the quietest possible
way for this file to be useless — so every `do` and `lead` is cross-checked
against the real paste, and the `when` vocabulary is closed.

Nothing consumes the book yet; these tests exist so it cannot rot between
authoring and wiring.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.local_battle import parse_showdown_team

HERE = Path(__file__).parent.parent / "showdown"
FIELDS = {"snow", "sun", "rain", "sand", "electricterrain", "grassyterrain",
          "psychicterrain", "mistyterrain", "trickroom", "screens"}
WHEN_KEYS = {"active", "max_turn", "field_up", "field_down", "hp_at_least",
             "unused"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@pytest.fixture(scope="module")
def book():
    return json.loads((HERE / "opening_book.json").read_text())["books"]


def team_for(name):
    hits = glob.glob(str(HERE / "teams" / "*" / f"{name}.txt"))
    if not hits:
        pytest.skip(f"no paste on disk for {name}")
    mons = parse_showdown_team(Path(hits[0]).read_text())
    species = {norm(m["species"]) for m in mons}
    moves = {norm(x) for m in mons for x in m["moves"]}
    return species, moves


def test_the_book_is_not_empty(book):
    assert len(book) >= 3


def test_every_lead_is_on_its_team(book):
    for name, b in book.items():
        species, _ = team_for(name)
        assert norm(b["lead"]) in species, f"{name}: lead {b['lead']} not on team"


def test_every_step_names_a_real_move_or_switch(book):
    for name, b in book.items():
        species, moves = team_for(name)
        for i, st in enumerate(b["steps"]):
            do = norm(st["do"])
            if str(st["do"]).startswith("switch "):
                assert norm(st["do"][7:]) in species, f"{name} step {i}"
            else:
                assert do in moves, f"{name} step {i}: '{st['do']}' not on team"


def test_every_step_targets_a_mon_that_has_the_move(book):
    """`active: X` plus `do: Y` is only meaningful if X actually knows Y."""
    for name, b in book.items():
        hits = glob.glob(str(HERE / "teams" / "*" / f"{name}.txt"))
        if not hits:
            continue
        mons = {norm(m["species"]): {norm(x) for x in m["moves"]}
                for m in parse_showdown_team(Path(hits[0]).read_text())}
        for i, st in enumerate(b["steps"]):
            act = st["when"].get("active")
            do = norm(st["do"])
            if act and not str(st["do"]).startswith("switch "):
                assert do in mons[norm(act)], (
                    f"{name} step {i}: {act} does not know {st['do']}")


def test_condition_vocabulary_is_closed(book):
    for name, b in book.items():
        for i, st in enumerate(b["steps"]):
            unknown = set(st["when"]) - WHEN_KEYS
            assert not unknown, f"{name} step {i}: unknown keys {unknown}"
            for k in ("field_up", "field_down"):
                if k in st["when"]:
                    assert st["when"][k] in FIELDS, f"{name} step {i}: {st['when'][k]}"
            if "hp_at_least" in st["when"]:
                assert 0 < st["when"]["hp_at_least"] <= 1.0
            if "max_turn" in st["when"]:
                assert st["when"]["max_turn"] >= 1


def test_every_step_has_a_rationale(book):
    for name, b in book.items():
        for i, st in enumerate(b["steps"]):
            assert len(st.get("why", "")) > 40, f"{name} step {i} lacks a why"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
