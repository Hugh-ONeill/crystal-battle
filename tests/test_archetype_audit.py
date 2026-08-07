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

from showdown.archetype_audit import audit, _off_role_species, _structure_defects

TEAMS = Path(__file__).parent.parent / "showdown" / "teams"

# OPPONENT PROXIES ARE EXEMPT, and deliberately so. These dirs are generated
# by build_opponent_proxy.py to REPLICATE a scouted opponent's actual teams,
# so their incoherences are FIDELITY, not defects — the LLM ladder bots
# genuinely run Araquanid without Sticky Web and Psychic Surge alongside the
# priority moves it self-disables, and a proxy that "fixed" those would be a
# worse model of the opponent we are trying to practise against. The gate
# exists for teams WE play, where an archetype claim not matching the paste
# is a lie about our own roster.
_PROXY_DIRS = ("rw_proxy", "ladder_proxy_", "ladder_field", "rw_fidelity")


def _is_proxy(p: Path) -> bool:
    return any(d in p.parent.name for d in _PROXY_DIRS)


@pytest.mark.parametrize("path", sorted(
    str(p) for p in TEAMS.glob("*/*.txt") if not _is_proxy(p)))
def test_team_contains_what_it_claims(path):
    ok, _label, problems = audit(path)
    # Harvest-defect findings (dead Choice/AV slots) REPORT, they don't
    # gate: benched/frozen dirs keep their historical pastes on purpose —
    # suite_v1 is frozen by policy and pool_hl_benched preserves retired
    # teams as played. Active pools are kept clean operationally (the
    # 2026-08-04 sweep repaired every live team), which the audit CLI
    # verifies; the parametrized gate here is for archetype-claim lies.
    hard = [p for p in problems
            if "dead slot" not in p and "unusable slot" not in p]
    assert not hard, f"{Path(path).name}: " + "; ".join(hard)


def test_structure_lint_catches_merged_blocks(tmp_path):
    """pool_hl/21_cinderace carried Clefable+Cinderace as ONE 8-move mon
    for weeks: the parser drops an unrecognized mid-block species line and
    piles the second mon's moves onto the first, so the team fielded 5 mons
    and every legality check passed. The lint must fire on both symptoms."""
    p = tmp_path / "99_merged.txt"
    p.write_text(
        "Clefable @ Leftovers\n"
        "Ability: Magic Guard\n"
        "Tera Type: Water\n"
        "EVs: 252 HP / 252 Def / 4 SpD\n"
        "Bold Nature\n"
        "- Moonblast\n"
        "- Soft-Boiled\n"
        "- Calm Mind\n"
        "- Knock Off\n"
        "Cinderace @ Heavy-Duty Boots\n"   # no blank separator: merged
        "Ability: Libero\n"
        "Tera Type: Fire\n"
        "EVs: 252 Atk / 4 SpD / 252 Spe\n"
        "Jolly Nature\n"
        "- Pyro Ball\n"
        "- U-turn\n"
        "- Court Change\n"
        "- Sucker Punch\n")
    ok, _label, problems = audit(str(p))
    assert not ok
    assert any("8 moves" in x for x in problems)
    assert any("not 6" in x for x in problems)


def test_structure_lint_duplicates_and_missing_ability():
    mons = [{"species": f"mon{i}", "ability": "ab",
             "moves": ["m1", "m2", "m3", "m4"]} for i in range(6)]
    mons[0]["moves"] = ["voltswitch", "voltswitch", "m3", "m4"]
    mons[1]["ability"] = None
    probs = _structure_defects(mons)
    assert any("duplicate move" in p and "voltswitch" in p for p in probs)
    assert any("no Ability" in p for p in probs)
    assert not any("not 6" in p for p in probs)
    assert not _structure_defects(
        [{"species": f"mon{i}", "ability": "ab",
          "moves": ["m1", "m2", "m3", "m4"]} for i in range(6)])


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
