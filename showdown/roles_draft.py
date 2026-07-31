#!/usr/bin/env python3
"""Draft role evidence for roles.json from grounded-rag, species-filtered.

WHY THIS EXISTS. A naive question to /retrieve returns TOPICALLY similar text,
which for role questions is routinely the wrong Pokemon: asking about Great
Tusk's Rapid Spin returned a Deoxys-Speed hazard-lead passage, and asking about
Dragonite's Multiscale/Boots returned Zamazenta's Boots set (2026-07-31). Both
are perfectly good passages about the topic and useless as evidence for the
species you asked about. Filtering by format alone does not fix it — the format
was already right in both misses.

So: retrieve, then keep only passages whose SOURCE names the species you asked
about, in the format you asked about. `source` looks like
    smogon#Great Tusk (gen9ou) - Defensive
which carries both, so the filter is exact rather than a similarity guess. The
API has no `k` parameter (verified against /openapi.json — question, corpus,
history only), hence filtering client-side rather than asking for more hits.

Output is for HUMAN REVIEW before anything reaches roles.json: an entry whose
evidence cannot be traced to a real passage is a guess, and the whole point of
the file is that guesses do not ship.

Usage:
  .venv/bin/python showdown/roles_draft.py "Great Tusk" "Corviknight"
  .venv/bin/python showdown/roles_draft.py --format gen9ou --show 2 Rillaboom
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8001/retrieve"
CHAOS = "gen9ou_chaos.json"

# Query phrasing decides whether you get analysis prose at all. A generic
# "what is X's role on a team" reads as a recommender request and routes to the
# TOOL corpora — it came back with tool#recommend_teammates and
# gen9ou_chaos#usage_rankings and zero set analyses. Naming concrete mechanics
# (the moves/ability it actually runs) pulls the written sets instead. So the
# question is built from this species' OWN top moves per the chaos stats, which
# also keeps it species-specific rather than topic-specific.
#
# `corpus="smogon"` does NOT do what it looks like (it did not restrict to the
# smogon corpus), so filtering is client-side on the source prefix.
ROLE_QUESTION = ("{species} {fmt} {moves} {ability} set analysis: what does "
                 "{species} do for its team and why does it run that item?")

# One query returns ONE fragment. Analyses are chunked and a top-30 Pokemon
# carries several sets across several chunks — Great Tusk has 5 chunks / ~4.5k
# characters spanning Offensive Utility, Defensive and Bulk Up, and a single
# query surfaced ~300 characters of that. Sweeping several angles and deduping
# by passage id recovers the whole analysis instead of whichever chunk happened
# to rank first.
ANGLES = [
    "{species} {fmt} {moves} {ability} set analysis: what does it do for its team?",
    "{species} {fmt} checks and counters: how is it beaten?",
    "{species} {fmt} teammates and team support",
    "{species} {fmt} defensive set: bulk, recovery, what it walls",
    "{species} {fmt} offensive set: setup, coverage, what it breaks",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def retrieve(question: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_CHAOS_CACHE: dict | None = None


def chaos_anchor(species: str) -> tuple[str, str]:
    """This species' own top moves + ability, to anchor the query on concrete
    mechanics. Returns ("", "") when the species is unknown — the query still
    works, just less sharply."""
    global _CHAOS_CACHE
    if _CHAOS_CACHE is None:
        try:
            from pathlib import Path
            raw = json.loads((Path(__file__).parent / CHAOS).read_text())
            data = raw.get("data", raw)
            _CHAOS_CACHE = {norm(k): v for k, v in data.items()}
        except Exception:
            _CHAOS_CACHE = {}
    e = _CHAOS_CACHE.get(norm(species))
    if not e:
        return "", ""
    moves = sorted((e.get("Moves") or {}).items(), key=lambda kv: -kv[1])[:3]
    abil = sorted((e.get("Abilities") or {}).items(), key=lambda kv: -kv[1])[:1]
    return " ".join(m for m, _ in moves), (abil[0][0] if abil else "")


def species_passages(species: str, fmt: str = "gen9ou") -> list[dict]:
    """Every distinct chunk of this species' analysis, best first.

    Sweeps several query angles and dedupes by passage id, because one query
    returns one fragment of a chunked, multi-set analysis.
    """
    moves, ability = chaos_anchor(species)
    want = norm(species)
    seen: dict = {}
    for tmpl in ANGLES:
        try:
            d = retrieve(tmpl.format(species=species, fmt=fmt,
                                     moves=moves, ability=ability))
        except Exception:
            continue                      # one bad angle must not lose the rest
        for p in d.get("passages") or []:
            src = str(p.get("source", ""))
            if not src.startswith("smogon#") or fmt not in src:
                continue
            # source: "smogon#<Species> (<fmt>) - <SetName>" — match the part
            # before the format tag so "Great Tusk" cannot match "Iron Treads"
            head = src.split("(")[0].split("#", 1)[-1]
            if norm(head) != want:
                continue
            key = p.get("id") or (src, (p.get("content") or "")[:80])
            seen.setdefault(key, p)
    hits = list(seen.values())
    hits.sort(key=lambda p: -(p.get("rerank_score") or 0))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="+")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--show", type=int, default=99, help="passages per species")
    ap.add_argument("--chars", type=int, default=420)
    ap.add_argument("--dump", help="write full chunk text for every species here")
    args = ap.parse_args()

    dump = open(args.dump, "w") if args.dump else None
    missing = []
    for sp in args.species:
        try:
            hits = species_passages(sp, args.format)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"\n=== {sp}: retrieval unavailable ({e}) — is grounded-rag up "
                  f"on :8001?", file=sys.stderr)
            continue
        if not hits:
            missing.append(sp)
            print(f"\n=== {sp} [{args.format}] — NO SPECIES-MATCHED PASSAGE")
            print("    Do not invent evidence for this one; either the corpus "
                  "lacks an analysis for it or the query missed. Leave the "
                  "entry marked unmeasured.")
            continue
        sets = sorted({h["source"].split("—")[-1].strip() for h in hits})
        print(f"\n=== {sp} [{args.format}] — {len(hits)} chunk(s), "
              f"{sum(len(h['content']) for h in hits)} chars, sets: {', '.join(sets)}")
        if dump:
            dump.write(f"\n\n########## {sp} [{args.format}] "
                       f"— {len(hits)} chunks, sets: {', '.join(sets)}\n")
            for h in hits:
                dump.write(f"\n--- [{h.get('source','?')}]\n"
                           f"{re.sub(chr(92)+'s+', ' ', h.get('content',''))}\n")
            dump.flush()
        for p in hits[: args.show]:
            txt = re.sub(r"\s+", " ", p.get("content", ""))
            print(f"  [{p.get('source','?')}]  rerank {p.get('rerank_score', 0):.3f}")
            print(f"    {txt[:args.chars]}...")
    if dump:
        dump.close()
        print(f"\n  full text written to {args.dump}", file=sys.stderr)
    if missing:
        print(f"\n  no evidence found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
