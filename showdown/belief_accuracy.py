#!/usr/bin/env python3
"""Per-game opponent-belief accuracy, tier vs tier, scored at TEAM PREVIEW.

WHY (2026-07-30): the team-archive matcher was shelved after three nulls, with
one explicit revisit condition — "only on-ladder where opponents' exact teams
ARE archive members". richwoman is foul-play (chat-signature ID) and runs a
small repeating pool: 11 of her 12 rosters are EXACT archive matches, covering
92% of her observed games. That precondition is now measured rather than
assumed, so the belief tiers deserve one more comparison — but an HONEST one.

The earlier measurement scored suite mirrors, and a naive rerun over the
scouting book scores the wrong thing: the book aggregates moves per species
ACROSS games, so richwoman's Ting-Lu shows 7 distinct moves and no single
4-move set can exceed 4/7 recall. This scores PER GAME against that game's
own reveals, which is the quantity the search actually needs.

Preview-time (zero reveals) is the headline comparison because it is
apples-to-apples — no tier gets to filter on observations — and it is exactly
where foul-play's archive advantage was measured (57.6% move accuracy).

READ THE RESULT CAREFULLY: belief ACCURACY IS NOT belief UTILITY. The archive
already beat fp on accuracy (67.5 vs 57.6) and still LOST the winrate gate,
because confidently-wrong joint details play worse than calibrated spread.
A win here justifies a paired live A/B; it does not justify deployment.

Usage:
  .venv/bin/python showdown/belief_accuracy.py --opponent richwoman
  .venv/bin/python showdown/belief_accuracy.py --opponent richwoman --max-cands 120
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))   # runnable as a script or a module


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# ---------------------------------------------------------------- game parsing


def parse_games(opponent: str, me: str = "PAC-Crystal"):
    """Per game: their 6 preview species + what THAT game actually revealed."""
    games = []
    for path in sorted(glob.glob(str(HERE / "bench" / "overnight_*_ladder.log"))):
        cur = None
        st = {}
        for raw in open(path, errors="replace"):
            m = re.search(r">(battle-gen9oulongtimer-\d+)", raw)
            if m:
                cur = m.group(1)
                st.setdefault(cur, dict(us=None, opp=None, side=None,
                                        preview=[], rev=defaultdict(
                                            lambda: {"moves": set(), "item": None,
                                                     "tera": None})))
            if cur is None or "|" not in raw:
                continue
            g = st[cur]
            p = raw[raw.find("|"):].rstrip("\n").split("|")[1:]
            if not p:
                continue
            tag = p[0]
            try:
                if tag == "player" and len(p) > 2:
                    if p[2].strip() == me:
                        g["us"] = p[1]
                    elif p[2].strip():
                        g["opp"], g["side"] = p[2].strip(), p[1]
                elif tag == "poke" and len(p) > 2 and g["side"] and p[1] == g["side"]:
                    g["preview"].append(p[2].split(",")[0].strip())
                elif tag == "move" and len(p) > 2 and g["side"] and \
                        p[1].startswith(g["side"] + "a"):
                    sp = p[1].split(": ", 1)[-1]
                    g["rev"][sp]["moves"].add(norm(p[2]))
                elif tag in ("-item", "-enditem") and len(p) > 2 and g["side"] and \
                        p[1].startswith(g["side"] + "a"):
                    sp = p[1].split(": ", 1)[-1]
                    g["rev"][sp]["item"] = norm(p[2])
                elif tag == "-terastallize" and len(p) > 2 and g["side"] and \
                        p[1].startswith(g["side"] + "a"):
                    sp = p[1].split(": ", 1)[-1]
                    g["rev"][sp]["tera"] = norm(p[2])
                elif tag == "win" and len(p) > 1:
                    g["won"] = p[1].strip() == me
            except Exception:
                pass
        for tag, g in st.items():
            if g["opp"] == opponent and len(g["preview"]) == 6 and g["rev"]:
                games.append(g)
    return games


# ---------------------------------------------------------------- belief tiers


class ArchiveTier:
    """Exact-roster match into the 208k metamon corpus; average over copies."""

    name = "archive"

    def __init__(self, max_cands=80):
        from showdown.team_archive import roster_key
        self.roster_key = roster_key
        idx = json.loads((HERE / "teams" / "team_archive_gen9ou.json").read_text())
        self.by, self.teams, self.root = idx["by_roster"], idx["teams"], Path(idx["root"])
        self.max_cands = max_cands
        self._cache: dict = {}

    def _parse(self, path):
        if path in self._cache:
            return self._cache[path]
        out, cur, moves, item, tera = {}, None, set(), None, None
        try:
            text = (self.root / path).read_text(errors="replace")
        except Exception:
            self._cache[path] = {}
            return {}
        for line in text.split("\n"):
            t = line.strip()
            if not t:
                if cur:
                    out[norm(cur)] = (moves, item, tera)
                cur, moves, item, tera = None, set(), None, None
                continue
            if cur is None:
                cur = t.split("@")[0].split("(")[0].strip()
                item = norm(t.split("@")[1]) if "@" in t else None
            elif t.startswith("- "):
                moves.add(norm(t[2:]))
            elif t.lower().startswith("tera type:"):
                tera = norm(t.split(":", 1)[1])
        if cur:
            out[norm(cur)] = (moves, item, tera)
        self._cache[path] = out
        return out

    def candidates(self, preview):
        k = self.roster_key(preview)
        idxs = self.by.get(k)
        if not idxs:
            return []
        out = []
        for ci in idxs[: self.max_cands]:
            entry = self.teams[ci]
            path = entry if isinstance(entry, str) else entry[0]
            t = self._parse(path)
            if t:
                out.append((t, 1.0))
        return out


class BookWeightedArchiveTier(ArchiveTier):
    """Archive candidates SELECTED by the opponent's own observed history.

    The plain archive knows her (95% BEST) but a blind draw gets 69% — a
    SELECTION problem. In-game `consistent()` filtering only helps once she has
    revealed something, yet belief is worth most at preview. Her history is
    available before turn 1: across prior games we have seen which moves/items/
    teras she actually runs on each species, so archive candidates that match
    that history are far likelier to be the team in front of us.

    Scored LEAVE-ONE-GAME-OUT — the profile for a game is built from her OTHER
    games only, never the one being scored, or this measures memorisation.
    """

    name = "archive+book"

    def __init__(self, max_cands=80, keep_frac=0.15, min_keep=3):
        super().__init__(max_cands)
        self.keep_frac, self.min_keep = keep_frac, min_keep
        self.hist = None      # species -> observed move set (LOO, set per game)

    def candidates(self, preview):
        cands = super().candidates(preview)
        if not cands or not self.hist:
            return cands
        scored = []
        for team, w in cands:
            hit = tot = 0.0
            for sp, (moves, item, tera) in team.items():
                seen = self.hist.get(sp)
                if not seen:
                    continue
                # moves she has been seen using on this species
                hit += len(moves & seen["moves"]); tot += len(moves)
                # item and tera are single-valued and high-signal: weight them
                # like a move slot each, so selection can act on the axes where
                # the archive's ceiling is highest (item 97%, tera 91%)
                if seen["items"]:
                    tot += 1.0
                    if item in seen["items"]:
                        hit += 1.0
                if seen["teras"]:
                    tot += 1.0
                    if tera in seen["teras"]:
                        hit += 1.0
            scored.append(((hit / tot) if tot else 0.0, team, w))
        scored.sort(key=lambda x: -x[0])
        k = max(self.min_keep, int(len(scored) * self.keep_frac))
        return [(t, w) for _, t, w in scored[:k]]


class PSTier:
    """Today's tier-1: PS's curated sets, per species, weight-averaged."""

    name = "curated-PS"

    def __init__(self):
        from showdown.ps_sets import get_index
        self.idx = get_index("gen9ou")

    def candidates(self, preview):
        # a "team" here is the cross-product view: per species independently
        per = {}
        for sp in preview:
            cands = (self.idx.candidates.get(norm(sp), []) if self.idx else [])
            if cands:
                per[norm(sp)] = [(set(c["moves"]), c.get("item"),
                                  c.get("tera_type"), c.get("weight", 1.0))
                                 for c in cands]
        return [(per, None)] if per else []


class ChaosTier:
    """Usage marginals: the top-4 moves / top item / top tera per species."""

    name = "chaos"

    def __init__(self):
        d = json.loads((HERE / "gen9ou_chaos.json").read_text())
        self.data = d.get("data", d)
        self.map = {norm(k): v for k, v in self.data.items()}

    def candidates(self, preview):
        per = {}
        for sp in preview:
            e = self.map.get(norm(sp))
            if not e:
                continue
            mv = sorted(e.get("Moves", {}).items(), key=lambda kv: -kv[1])[:4]
            it = sorted(e.get("Items", {}).items(), key=lambda kv: -kv[1])[:1]
            tt = sorted(e.get("Tera Types", {}).items(), key=lambda kv: -kv[1])[:1]
            per[norm(sp)] = [({norm(m) for m, _ in mv},
                              norm(it[0][0]) if it else None,
                              norm(tt[0][0]) if tt else None, 1.0)]
        return [(per, None)] if per else []


# ---------------------------------------------------------------- scoring


def score(tier, games):
    """Two numbers per axis, because they answer different questions.

    MEAN = expected recall of a RANDOMLY DRAWN candidate. This is what the
    search actually experiences, since each sampled world draws one candidate.
    BEST = recall of the single best candidate available. This asks whether the
    tier CONTAINS her real set at all — if BEST is high while MEAN is low, the
    information is present and the problem is selection (which revealed-move
    filtering fixes as a game progresses); if BEST is low the tier simply does
    not know her.

    Comparing MEAN across tiers is only fair between tiers that sample. The
    chaos tier here is built as a single modal set (top-4 moves / top item /
    top tera), so its MEAN and BEST coincide and it is effectively scored as an
    ORACLE-of-the-mode — flattering versus tiers averaged over dozens of
    diverse candidates. Read chaos's column as "how good is the single most
    likely set", not as a like-for-like sampling comparison.
    """
    mv_hit = mv_tot = 0.0
    mv_best = 0.0
    it_hit = it_tot = 0.0
    it_best = 0.0
    tr_hit = tr_tot = 0.0
    tr_best = 0.0
    covered = 0
    for gi, g in enumerate(games):
        # leave-one-game-out history for any tier that consumes it
        if hasattr(tier, "hist"):
            h = defaultdict(lambda: {"moves": set(), "items": set(), "teras": set()})
            for j, o in enumerate(games):
                if j == gi:
                    continue
                for sp, rev in o["rev"].items():
                    e = h[norm(sp)]
                    e["moves"] |= rev["moves"]
                    if rev["item"]:
                        e["items"].add(rev["item"])
                    if rev["tera"]:
                        e["teras"].add(rev["tera"])
            tier.hist = h
        cands = tier.candidates(g["preview"])
        if not cands:
            continue
        covered += 1
        for sp, rev in g["rev"].items():
            n = norm(sp)
            # per-species candidate list, whichever shape the tier returned
            if isinstance(cands[0][0], dict) and cands[0][1] is None:
                opts = cands[0][0].get(n)
                if not opts:
                    continue
                wsum = sum(o[3] for o in opts) or 1.0
                pm = sum(o[3] * len(rev["moves"] & o[0]) for o in opts) / wsum
                pi = sum(o[3] for o in opts if rev["item"] and o[1] == rev["item"]) / wsum
                pt = sum(o[3] for o in opts if rev["tera"] and o[2] == rev["tera"]) / wsum
                bm = max(len(rev["moves"] & o[0]) for o in opts)
                bi = max((1 if rev["item"] and o[1] == rev["item"] else 0) for o in opts)
                bt = max((1 if rev["tera"] and o[2] == rev["tera"] else 0) for o in opts)
            else:
                sets = [t.get(n) for t, _ in cands if t.get(n)]
                if not sets:
                    continue
                pm = sum(len(rev["moves"] & s[0]) for s in sets) / len(sets)
                pi = sum(1 for s in sets if rev["item"] and s[1] == rev["item"]) / len(sets)
                pt = sum(1 for s in sets if rev["tera"] and s[2] == rev["tera"]) / len(sets)
                bm = max(len(rev["moves"] & s[0]) for s in sets)
                bi = max((1 if rev["item"] and s[1] == rev["item"] else 0) for s in sets)
                bt = max((1 if rev["tera"] and s[2] == rev["tera"] else 0) for s in sets)
            if rev["moves"]:
                mv_hit += pm; mv_best += bm; mv_tot += len(rev["moves"])
            if rev["item"]:
                it_hit += pi; it_best += bi; it_tot += 1
            if rev["tera"]:
                tr_hit += pt; tr_best += bt; tr_tot += 1
    pct = lambda a, b: 100 * a / b if b else 0.0
    return dict(games=covered,
                move=pct(mv_hit, mv_tot), move_best=pct(mv_best, mv_tot),
                item=pct(it_hit, it_tot), item_best=pct(it_best, it_tot),
                tera=pct(tr_hit, tr_tot), tera_best=pct(tr_best, tr_tot),
                n_move=int(mv_tot), n_item=int(it_tot), n_tera=int(tr_tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="richwoman")
    ap.add_argument("--max-cands", type=int, default=80,
                    help="archive copies averaged per roster (speed cap)")
    args = ap.parse_args()

    games = parse_games(args.opponent)
    print(f"{len(games)} games vs {args.opponent} with a full preview roster "
          f"and at least one reveal\n")
    if not games:
        return

    top1 = BookWeightedArchiveTier(args.max_cands, keep_frac=0.0, min_keep=1)
    top1.name = "arch+book@1"   # single best-scoring candidate = a MODAL belief,
                                # the like-for-like comparison against chaos
    tiers = [ArchiveTier(args.max_cands),
             BookWeightedArchiveTier(args.max_cands),
             top1, PSTier(), ChaosTier()]
    print(f"  {'tier':12s} {'games':>6s} | {'move mean':>10s} {'move BEST':>10s} "
          f"| {'item mean':>10s} {'item BEST':>10s} | {'tera mean':>10s} {'tera BEST':>10s}")
    for t in tiers:
        r = score(t, games)
        print(f"  {t.name:12s} {r['games']:6d} | {r['move']:9.1f}% {r['move_best']:9.1f}% "
              f"| {r['item']:9.1f}% {r['item_best']:9.1f}% "
              f"| {r['tera']:9.1f}% {r['tera_best']:9.1f}%")
    r = score(tiers[0], games)
    print(f"\n  (n = {r['n_move']} revealed moves / {r['n_item']} items / "
          f"{r['n_tera']} teras)")
    print("  MEAN = expected recall of a randomly drawn candidate (what a sampled"
          "\n         world actually gets).  BEST = the best candidate available"
          "\n         (does the tier CONTAIN her set at all?).")
    print("  CAVEAT: the chaos tier is a single MODAL set, so its mean==best and it"
          "\n         is scored as an oracle-of-the-mode — not a like-for-like"
          "\n         sampling comparison against tiers averaged over many candidates.")
    print("  Scored at TEAM PREVIEW (zero reveals used) against each game's own"
          "\n  reveals. ACCURACY IS NOT UTILITY — a win justifies a paired live"
          "\n  A/B, not deployment.")


if __name__ == "__main__":
    main()
