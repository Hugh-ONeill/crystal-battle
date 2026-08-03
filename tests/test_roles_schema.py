"""roles.json must stay consistent with its own documented schema.

The file's header comment IS its specification, and it drifted twice before
2026-08-02: value_curve documented four values while six were in use, and
`review` was undocumented entirely. Worse, the tag vocabulary had grown
three spellings of one role (hazard-removal / hazard-remover /
hazard-control), so a consumer keying on any single spelling silently
missed most hazard removers — and tags are the documented consumer key.

These tests fail when the data grows a value the header does not document,
which is the cheap moment to fix it.
"""
import json
import re
from pathlib import Path

import pytest

ROLES = Path(__file__).parent.parent / "showdown" / "roles.json"


@pytest.fixture(scope="module")
def doc():
    return json.loads(ROLES.read_text())


@pytest.fixture(scope="module")
def header(doc):
    return "\n".join(doc["_comment"])


def entries(doc):
    return doc["roles"]


def holders(doc):
    """every dict that may carry tags/axis: species entries and their sets."""
    for sp, e in entries(doc).items():
        yield sp, e
        for s in e.get("sets") or []:
            yield f"{sp}:{s.get('name')}", s


def test_every_value_curve_is_documented(doc, header):
    used = {e["value_curve"] for e in entries(doc).values()
            if e.get("value_curve")}
    undocumented = {v for v in used if v not in header}
    assert not undocumented, (
        f"value_curve values missing from the header: {undocumented}")


def test_every_review_class_is_documented(doc, header):
    used = {e["review"] for e in entries(doc).values() if e.get("review")}
    assert used, "no review tags at all"
    assert not {v for v in used if v not in header}


def test_every_entry_condition_is_documented(doc, header):
    used = {e["entry_condition"] for e in entries(doc).values()
            if e.get("entry_condition")}
    assert not {v for v in used if v not in header}


def test_every_tag_is_in_the_documented_vocabulary(doc, header):
    used = {t for _, h in holders(doc) for t in (h.get("tags") or [])}
    undocumented = {t for t in used if not re.search(rf"\b{re.escape(t)}\b",
                                                     header)}
    assert not undocumented, (
        f"tags used but absent from the header vocabulary: {undocumented} — "
        "add them to the VOCABULARY block when first used")


def test_no_resurrected_spelling_variants(doc):
    """The 2026-08-02 canonicalization must not regress."""
    dead = {"hazard-removal", "hazard-control"}
    for who, h in holders(doc):
        assert not (dead & set(h.get("tags") or [])), (
            f"{who} uses a merged tag spelling; canonical is hazard-remover")


def test_closed_enums(doc):
    for sp, e in entries(doc).items():
        assert e.get("preserve") in (None, "low", "med", "high"), sp
        assert e.get("lead_intent") in (None, "avoid", "neutral", "strong"), sp
    for who, h in holders(doc):
        for k, v in (h.get("axis") or {}).items():
            assert k in ("attacks", "defends"), who
            assert v in ("physical", "special", "mixed"), who
        assert h.get("deployment") in (
            None, "lead", "bait-switch", "pivot-cycle", "late-cleaner",
            "sacrifice", "setup-window"), who


def test_required_fields_and_prevalence_sanity(doc):
    for sp, e in entries(doc).items():
        assert e.get("tags") and e.get("fact") and e.get("review"), sp
        assert e.get("provenance"), sp
        for s in e.get("sets") or []:
            assert s.get("name"), sp
            # a residual "Other / unnamed" placeholder exists to show the
            # uncovered share, so it legitimately claims no role
            if s.get("prevalence_method") != "residual":
                assert s.get("tags"), f"{sp}:{s['name']}"
            p = s.get("prevalence")
            if p is not None:
                assert 0 < p <= 1, f"{sp}:{s['name']} prevalence {p}"
                assert s.get("prevalence_method") in (
                    "signature", "residual", "estimate"), sp


def test_dossier_exposes_ability_fields(doc):
    """The overlay dossier must render ability info.

    It did not until 2026-08-02: `_ENTRY_FIELDS` omitted both `ability` and
    `ability_split`, so 89 entries' abilities were invisible to the LLM —
    which then cited `okidogi.ability` 18 times, its most-cited path, for a
    species rendered as "(no entry)". Regression guard for that omission.
    """
    from showdown.overlay import _ENTRY_FIELDS
    assert "ability" in _ENTRY_FIELDS and "ability_split" in _ENTRY_FIELDS


def test_ability_citation_resolves_against_a_split(doc):
    """`X.ability` must resolve when the entry records an ability_split."""
    from showdown.overlay import OverlayShadow
    sh = OverlayShadow.__new__(OverlayShadow)
    sh.roles = doc["roles"]
    split = [sp for sp, e in doc["roles"].items() if e.get("ability_split")]
    assert split, "no ability_split entries to test against"
    assert sh._rule_resolves(f"{split[0]}.ability")
    assert not sh._rule_resolves(f"{split[0]}.not_a_field")


def test_engine_blind_is_consumer_facing(doc):
    """What the SEARCH cannot model must reach consumers, not sit in review-only
    text. Before 2026-08-02 the Illusion explanation — the single most
    decision-relevant fact about Zoroark-H — lived in `provenance`, which no
    consumer ever sees, while `fact` merely pointed at it."""
    from showdown.overlay import _ENTRY_FIELDS
    assert "engine_blind" in _ENTRY_FIELDS
    blind = {sp for sp, e in entries(doc).items() if e.get("engine_blind")}
    assert {"zoroarkhisui", "gengar"} <= blind, (
        "the known unmodelled abilities (Illusion, Cursed Body) must be "
        "declared where a consumer can read them")


def test_no_fact_dangles_into_review_only_text(doc):
    """A `fact` may not defer to text the consumer cannot see."""
    import re
    for sp, e in entries(doc).items():
        f = e.get("fact", "")
        assert not re.search(r",\s*(below|above)\.\s*$", f), sp
        assert "see the provenance" not in f.lower(), sp


def test_every_entry_is_signed_off_and_provenance_is_not_stale(doc):
    """The file was signed off 2026-08-02, clearing the header's review gate."""
    for sp, e in entries(doc).items():
        assert e.get("reviewed_on"), f"{sp} has no reviewed_on"
        assert "not yet user-reviewed" not in e["provenance"].lower(), sp


def test_review_class_survives_review(doc):
    """`review` records WHERE A CLAIM CAME FROM and must not be flattened to
    user-corrected just because a human approved the file — only entries whose
    claims a human actually changed carry that class."""
    classes = {e["review"] for e in entries(doc).values()}
    assert {"rag-grounded", "usage-only"} <= classes, (
        "provenance classes were destroyed by the review pass")


def test_ability_cited_by_name_resolves(doc):
    """Once abilities became visible in the dossier the model started citing
    them BY VALUE — `gholdengo.goodasgold`, 11 times in the first session —
    rather than by field name. That names a real fact in a real entry, so
    discarding it would penalise the model for a convention it was never
    told."""
    from showdown.overlay import OverlayShadow
    sh = OverlayShadow.__new__(OverlayShadow)
    sh.roles = doc["roles"]
    named = [(sp, e["ability"]) for sp, e in doc["roles"].items()
             if e.get("ability")]
    assert named
    sp, ab = named[0]
    assert sh._rule_resolves(f"{sp}.{ab}")
    assert not sh._rule_resolves(f"{sp}.notanability")
    split = [(sp, k) for sp, e in doc["roles"].items()
             for k in (e.get("ability_split") or {})]
    if split:
        sp, k = split[0]
        assert sh._rule_resolves(f"{sp}.{k}")


def test_usage_citations_resolve_now_that_the_dossier_carries_them(doc):
    """`X.moves` and `X.item` were the model's TOP unresolved citations for
    weeks (ragingbolt.item 23, blissey.moves 21, greattusk.moves 15). They
    are not roles fields, but the dossier now carries a per-species usage
    prior, so they reference something real and must be honoured."""
    from showdown.overlay import OverlayShadow
    sh = OverlayShadow.__new__(OverlayShadow)
    sh.roles = doc["roles"]
    assert sh._rule_resolves("ragingbolt.item")
    assert sh._rule_resolves("blissey.moves")
    # a species with no chaos entry still cannot be cited
    assert not sh._rule_resolves("notapokemon.moves")
    # and an invented field is still rejected
    assert not sh._rule_resolves("gliscor.nonsense")


def test_the_usage_prior_is_deflation_corrected(doc):
    """Raw chaos shares understate by ~2x; Gliscor is 99% Toxic Orb, not 45%."""
    from showdown.overlay import OverlayShadow
    line = OverlayShadow._usage_prior("gliscor")
    assert line and "toxicorb" in line
    pct = int(line.split("toxicorb")[1].split("%")[0].strip())
    assert pct >= 90, f"expected a corrected share, got {pct}%"


def test_the_dossier_shows_joint_sets_not_just_marginals(doc):
    """Marginals cannot express which moves go together. Gliscor lists seven
    moves above 25% for four slots, and nothing in that list says Swords
    Dance pairs with Facade while Spikes pairs with Toxic — two different
    Pokemon. The curated builds are the joint view."""
    from showdown.overlay import OverlayShadow
    line = OverlayShadow._usage_prior("gliscor")
    assert "set '" in line, "no curated builds rendered"
    sd = [l for l in line.split("\n") if "Swords Dance" in l]
    assert sd and "Facade" in sd[0], "the SD build must show its real partner"
    util = [l for l in line.split("\n") if "Spikes" in l]
    assert util and "Toxic" in util[0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
