"""Every team must contain the mechanic it claims — as a test, not a habit.

The webs family carried no Sticky Web for weeks and validated as LEGAL the
whole time, because legality cannot express "this team is not what it claims".
This pins the audit so the next one is caught by the suite instead of by
someone noticing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.archetype_audit import audit, _off_role_species

TEAMS = Path(__file__).parent.parent / "showdown" / "teams"


@pytest.mark.parametrize("path", sorted(str(p) for p in TEAMS.glob("*/*.txt")))
def test_team_contains_what_it_claims(path):
    ok, _label, problems = audit(path)
    assert ok, f"{Path(path).name}: " + "; ".join(problems)


def test_off_role_threshold_separates_defining_from_optional():
    """The criterion must be near-universal USAGE, not corpus presence: an
    earlier version keyed on 'every curated set carries it' and flagged
    Blissey for lacking Stealth Rock (60% of real sets) and Skarmory for
    lacking Spikes (79%) — both ordinary builds."""
    off = _off_role_species()
    assert "stickyweb" in off.get("araquanid", set())
    assert "stickyweb" in off.get("ribombee", set())
    assert "stealthrock" not in off.get("blissey", set())
    assert "spikes" not in off.get("skarmory", set())
    assert "stealthrock" not in off.get("irontreads", set())
