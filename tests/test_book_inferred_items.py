"""The scouting book must learn the items that never announce.

Choice items, Boots and Assault Vest emit no protocol event, so a book built
only from what the wire says can never record them — 13 of richwoman's 34
species had no item after 321 games for exactly this reason. Our own
inferences (speed floors, damage brackets, zero-chip entry) are the only
source, and they were being discarded at game end.

SAFETY: the book is TIER 0 and overlays every other set source, so a wrong
item persisted here becomes authoritative against that opponent forever.
The player therefore emits only inferences the FINAL evidence still
supports, and a direct sighting always outranks an inference.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import showdown.scouting_book as sb


def test_the_inferred_line_is_recognised():
    m = sb._INFERRED.search("|inferreditem|gliscor|toxicorb")
    assert m and m.group(1) == "gliscor" and m.group(2) == "toxicorb"


def test_a_from_tag_item_is_recognised():
    """The bug that left 22 of 34 species item-less: items announce as a
    [from] TAG on an ordinary event, not only on |-item| lines."""
    line = "|-status|p2a: Gliscor|tox|[from] item: Toxic Orb"
    m = sb._ITEM_TAG.search(line)
    assert m and m.group(3).strip() == "Toxic Orb"


def test_the_of_tag_reassigns_ownership():
    """Rocky Helmet chip names the DEFENDER's item on a line describing
    damage to the ATTACKER."""
    line = ("|-damage|p1a: Kingambit|84/100|[from] item: Rocky Helmet"
            "|[of] p2a: Corviknight")
    tag = sb._ITEM_TAG.search(line)
    of = sb._OF_TAG.search(line)
    assert tag.group(1) == "p1"          # the line is about our mon
    assert of and of.group(1) == "p2"    # the item is theirs


def test_player_filters_inferences_the_evidence_refutes():
    """A refuted inference must never reach the book — the Choice Band
    Gliscor would otherwise have become a permanent belief."""
    from showdown.set_inference import BattleObservations
    obs = BattleObservations()
    obs.confirmed["gliscor"] = "choiceband"
    obs.choice_disproven.add("gliscor")
    # the emit path drops it on either guard
    assert "gliscor" in obs.choice_disproven
    obs2 = BattleObservations()
    obs2.confirmed["corviknight"] = "leftovers"
    obs2._forbid("corviknight", "leftovers")
    assert "leftovers" in obs2.forbidden("corviknight")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
