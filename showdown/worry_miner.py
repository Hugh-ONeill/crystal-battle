#!/usr/bin/env python3
"""Mine the overlay shadow log's `worry` field — the both-worlds-wrong detector.

The worry field (required since 2026-08-01) asks the LLM to name the thing
most likely to invalidate the engine's choice. This tool answers two
questions over the accrued corpus:

  (a) WHAT does it worry about — taxonomy (dossier citations vs choice
      echoes vs prose diagnoses), themes, species, turns;
  (b) the EVIDENCE GATE for the LLM-authored-world idea: how often does a
      worry name a concrete opponent hypothesis (move/item/ability) that is
      ABSENT from every world's assumed set — i.e. a scenario the search
      structurally could not have priced. Verified against the record's own
      per-world `assumed_sets`, with the chaos file as the token vocabulary
      so prose like "Malignant Chain" is recognized as a move name.

Substring traps handled: tokens are matched LONGEST-FIRST with span
claiming ("thunderbolt" consumes its span so "thunder" cannot re-match it);
ids <= 6 normalized chars ("rest", "roar") additionally require raw word
boundaries. Attribution traps handled: a matched token is credited to the
NEAREST species mention in the text (so Raging Bolt's listed moves are not
blamed on the Zapdos named earlier in the sentence), and a token equal to
the engine's own chosen move is skipped ("the engine is choosing bodypress"
is not an opponent hypothesis — costs us the rare mirror-move worry).

  .venv/bin/python showdown/worry_miner.py
  .venv/bin/python showdown/worry_miner.py --hits 20 --min-share 0.05
"""

from __future__ import annotations

import argparse
import collections
import json
import re


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


CITATION_RE = re.compile(
    r"^\s*[a-z0-9-]+\.[a-z_]+(\s*[,;/\s]\s*[a-z0-9-]+\.[a-z_]+)*\s*\.?\s*$")
THEMES = {
    "world-split": re.compile(
        r"world\s*[01]|both worlds|either world|neither world|worlds? (dis)?agree",
        re.I),
    "near-tie": re.compile(
        r"near-?(tie|uniform)|low margin|margin is|coin-?flip|uncertain", re.I),
    "threat": re.compile(
        r"vulnerab|punish|revenge|threaten|risk|loses to|[o2]hko|sweep|set\s?up",
        re.I),
    "speed": re.compile(r"outspeed|faster|slower|scarf|speed tier", re.I),
}


def classify(txt: str) -> str:
    if CITATION_RE.match(txt):
        return "citation"
    body = re.sub(r"^engine_choice:?\s*", "", txt.strip(), flags=re.I)
    if body != txt.strip() and len(body.split()) <= 4:
        return "choice-echo"
    return "prose"


def species_vocab(chaos: dict, min_share: float):
    """(vocab, all_tokens): vocab[n_sp] = (display, {n_tok: (kind, disp)});
    all_tokens = the union across species, for longest-first masking."""
    vocab, all_tokens = {}, {}
    for sp, d in chaos["data"].items():
        raw = d.get("Raw count") or 1
        toks = {}
        for kind, key, thresh in (("move", "Moves", min_share),
                                  ("item", "Items", min_share),
                                  ("ability", "Abilities", 0.05)):
            for tok, cnt in (d.get(key) or {}).items():
                if tok in ("", "nothing") or cnt / raw < thresh:
                    continue
                toks[norm(tok)] = (kind, tok)
        vocab[norm(sp)] = (sp, toks)
        all_tokens.update(toks)
    return vocab, all_tokens


def species_probe(n_sp: str) -> str:
    return n_sp[:8]  # landorustherian -> "landorus"-T mentions still match


def find_spans(n_txt: str, needle: str, claimed: list) -> list:
    """occurrences of needle in n_txt not overlapping already-claimed spans."""
    out, i = [], 0
    while (i := n_txt.find(needle, i)) != -1:
        span = (i, i + len(needle))
        if not any(a < span[1] and span[0] < b for a, b in claimed):
            out.append(span)
        i += 1
    return out


def match_tokens(worry_raw: str, n_txt: str, all_tokens: dict) -> list:
    """[(n_tok, pos)] longest-first with span claiming + short-token guard."""
    claimed, out = [], []
    for n_tok in sorted(all_tokens, key=len, reverse=True):
        if len(n_tok) <= 6 and not re.search(
                r"\b" + n_tok + r"\b", worry_raw.lower()):
            continue
        for span in find_spans(n_txt, n_tok, claimed):
            claimed.append(span)
            out.append((n_tok, span[0]))
    return out


def modeled_for(rec: dict, n_sp: str):
    """union of (moves, item, ability) every world assumes for species."""
    moves, items, abils = set(), set(), set()
    for w in rec.get("worlds") or []:
        for sp, s in (w.get("assumed_sets") or {}).items():
            if norm(sp) != n_sp:
                continue
            moves |= {norm(m) for m in s.get("moves") or []}
            if s.get("item"):
                items.add(norm(s["item"]))
            if s.get("ability"):
                abils.add(norm(s["ability"]))
    return {"move": moves, "item": items, "ability": abils}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="showdown/overlay_shadow.jsonl")
    ap.add_argument("--chaos", default="showdown/gen9ou_chaos.json")
    ap.add_argument("--hits", type=int, default=15,
                    help="unmodeled-hypothesis hits to print")
    ap.add_argument("--min-share", type=float, default=0.03,
                    help="chaos usage share for a token to enter the vocab")
    args = ap.parse_args()

    vocab, all_tokens = species_vocab(json.load(open(args.chaos)),
                                      args.min_share)
    recs = []
    for line in open(args.log):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("llm"), dict) and r["llm"].get("worry"):
            recs.append(r)
    games = {r["tag"] for r in recs}

    tax = collections.Counter()
    themes = collections.Counter()
    cited = collections.Counter()
    named_sp = collections.Counter()
    by_phase = collections.Counter()
    by_game_prose = collections.Counter()
    prose_quotes = collections.defaultdict(list)
    hits, split_articulations = [], 0

    for r in recs:
        txt = r["llm"]["worry"].strip()
        kind = classify(txt)
        tax[kind] += 1
        t = r.get("turn") or 0
        by_phase["opening(T<=9)" if t <= 9 else
                 "midgame(10-19)" if t <= 19 else "lategame(20+)"] += 1
        if kind == "citation":
            for c in re.findall(r"[a-z0-9-]+\.[a-z_]+", txt):
                cited[c] += 1
            continue
        if kind != "prose":
            continue
        by_game_prose[r["tag"]] += 1
        prose_quotes[r["tag"]].append((t, txt))
        for name, rx in THEMES.items():
            if rx.search(txt):
                themes[name] += 1
        if re.search(r"world\s*0.*world\s*1|world\s*1.*world\s*0", txt,
                     re.I | re.S):
            split_articulations += 1

        n_txt = norm(txt)
        opp = set()
        for w in r.get("worlds") or []:
            opp |= {norm(sp) for sp in (w.get("assumed_sets") or {})}
        # every species mention (opp or not) is an attribution anchor
        anchors = []
        for n_sp in vocab:
            for pos, _ in find_spans(n_txt, species_probe(n_sp), []):
                anchors.append((pos, n_sp, n_sp in opp))
        for n_sp in {a[1] for a in anchors if a[2]}:
            named_sp[vocab[n_sp][0]] += 1
        our_move = norm(re.sub(r"^switch\s+\S+$", "",
                               r.get("engine_choice") or ""))
        for n_tok, pos in match_tokens(txt, n_txt, all_tokens):
            if n_tok == our_move:
                continue
            near = [(abs(pos - p), not is_opp, n_sp, is_opp)
                    for p, n_sp, is_opp in anchors]  # opp wins probe ties
            if not near:
                continue
            dist, _, n_sp, is_opp = min(near)
            if dist > 250 or not is_opp:
                continue
            sp_disp, toks = vocab[n_sp]
            if n_tok not in toks:
                continue
            kind, disp = toks[n_tok]
            if n_tok in modeled_for(r, n_sp)[kind]:
                continue
            hits.append({"tag": r["tag"], "turn": t, "species": sp_disp,
                         "kind": kind, "token": disp, "worry": txt,
                         "reasons": r.get("reasons"),
                         "choice": r.get("engine_choice"),
                         "margin": r.get("engine_margin")})

    n = len(recs)
    print(f"{n} worries / {len(games)} games   "
          f"phases: " + "  ".join(f"{k} {v}" for k, v in sorted(by_phase.items())))
    print("\n  taxonomy: " + "  ".join(
        f"{k} {v} ({v / n:.0%})" for k, v in tax.most_common()))
    print("  prose themes: " + "  ".join(
        f"{k} {v}" for k, v in themes.most_common()))
    print(f"  explicit world-0-vs-world-1 articulations: {split_articulations}")

    print("\n  top dossier fields cited as the worry itself:")
    for c, v in cited.most_common(10):
        print(f"    {v:3d}  {c}")
    print("\n  species named in prose worries:")
    for sp, v in named_sp.most_common(10):
        print(f"    {v:3d}  {sp}")

    uniq = {}
    for h in hits:  # one worry can hit several tokens; keep all, dedup exact
        uniq.setdefault((h["tag"], h["turn"], h["species"], h["token"]), h)
    hs = sorted(uniq.values(), key=lambda h: -(h["margin"] or 0))
    hit_games = {h["tag"] for h in hs}
    print(f"\n== UNMODELED-HYPOTHESIS HITS: {len(hs)} "
          f"(in {len(hit_games)}/{len(games)} games) — worry names a "
          f"move/item/ability NO world assumed ==")
    for h in hs[:args.hits]:
        print(f"\n  [{h['tag']} T{h['turn']}] {h['species']} {h['kind']} "
              f"'{h['token']}' unmodeled; engine {h['choice']} "
              f"margin {h['margin']}")
        print(f"    {h['worry'][:300]}")

    print("\n  most-worried games (prose count):")
    for tag, v in by_game_prose.most_common(5):
        t0, q = prose_quotes[tag][0]
        print(f"    {v:2d}  {tag}   e.g. T{t0}: {q[:150]}")


if __name__ == "__main__":
    main()
