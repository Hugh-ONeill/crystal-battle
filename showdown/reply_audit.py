#!/usr/bin/env python3
"""Reply-channel (§5.2) accuracy gate: join the shadow consults' reply
predictions to what the opponent ACTUALLY clicked, from the ladder logs.

This is the channel's phase-1 bar, learned from the world channel's death:
belief accuracy gets measured against ground truth BEFORE any re-solve or
A/B exists. Three policies are scored on every joined turn:

  gemma    the consult's `reply_pred` distribution (absent on pre-2026-08-06
           records — they log no prediction and only feed the baselines)
  engine   the search's own implied reply: each world's their_replies visit
           shares, equal-vote averaged (the distribution §5.2 proposes to
           replace — THE bar to beat)
  uniform  1/|options| floor

Ground truth: per (battle, turn), the opponent's first |move| or |switch|
after the |turn| line. Excluded as non-decisions: |drag| (phazed), a switch
following the opponent's own faint in the same block (forced replacement),
and turns where the collapsed option set has fewer than 2 entries (locked/
forced). Tera is collapsed ("x-tera" scores as "x"): tera timing is a
separate question from action prediction.

Usage:
  .venv/bin/python showdown/reply_audit.py [--shadow PATH] [--logs GLOB]
      [--name PAC-Crystal] [--matched-only]

--matched-only scores engine/uniform ONLY on turns carrying a gemma
prediction, so all three policies share a population. This is the read that
closed the channel (2026-08-09): the registered 33.9% bar was set on the
pre-build field (richwoman-heavy), gemma's consults landed on the post-08-06
field (74% MuratiBot-1), and on matched turns the engine scores 42.5% on its
own — gemma's headline 43.2% was the population, not the predictor.
"""

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def collapse(action: str) -> str:
    """'x-tera' -> 'x'; leaves 'switch x' alone."""
    return action[:-5] if action.endswith("-tera") else action


def parse_logs(paths, our_name: str):
    """-> ({(room, turn): action-or-None}, {room: opponent_name})"""
    decisions, opp_name = {}, {}
    sides = defaultdict(dict)          # room -> {p1: name, p2: name}
    room = None
    state = defaultdict(lambda: {"turn": 0, "decided": True,
                                 "opp_fainted": False})
    for path in paths:
        for line in open(path, errors="replace"):
            m = re.search(r">(battle-\S+)", line)
            if m:
                room = m.group(1)
                continue
            if room is None or "|" not in line:
                continue
            m = re.search(r"\|player\|(p[12])\|([^|]+)\|", line)
            if m:
                sides[room][m.group(1)] = m.group(2)
                if m.group(2) != our_name:
                    opp_name[room] = m.group(2)
                continue
            opp = next((p for p, n in sides[room].items()
                        if n != our_name), None)
            if opp is None:
                continue
            st = state[room]
            m = re.search(r"\|turn\|(\d+)", line)
            if m:
                st["turn"] = int(m.group(1))
                st["decided"] = False
                st["opp_fainted"] = False
                continue
            if re.search(r"\|faint\|" + opp + "a", line):
                st["opp_fainted"] = True
                continue
            if st["decided"] or st["turn"] == 0:
                continue
            m = re.search(r"\|move\|" + opp + r"a: [^|]*\|([^|\n]+)", line)
            if m:
                decisions[(room, st["turn"])] = _alnum(m.group(1))
                st["decided"] = True
                continue
            if re.search(r"\|drag\|" + opp + "a", line):
                st["decided"] = True          # phazed in: not a decision
                continue
            m = re.search(r"\|switch\|" + opp + r"a: [^|]*\|([^,|\n]+)", line)
            if m:
                st["decided"] = True
                if not st["opp_fainted"]:     # else: forced replacement
                    decisions[(room, st["turn"])] = \
                        "switch " + _alnum(m.group(1))
    return decisions, opp_name


def engine_dist(rec) -> dict:
    """Equal-vote average of per-world their_replies visit shares,
    tera-collapsed. This is the emission's top-6 per world — the same
    truncation for every record era, so comparisons stay fair."""
    acc = defaultdict(float)
    worlds = rec.get("worlds") or []
    n = 0
    for w in worlds:
        replies = w.get("their_replies") or []
        total = sum(v for _, v in replies)
        if total <= 0:
            continue
        n += 1
        for action, v in replies:
            acc[collapse(action)] += v / total
    return {k: v / n for k, v in acc.items()} if n else {}


def brier(dist: dict, actual: str, support: set) -> float:
    return sum((dist.get(o, 0.0) - (1.0 if o == actual else 0.0)) ** 2
               for o in support | {actual})


def decided(rec) -> bool:
    """True when the position is already decided in every world (top row
    q >= 0.95 or <= 0.05). There the opponent's replies all carry the same
    Q, decoupled UCT's visit marginal is amplified exploration noise, and
    'engine implied reply' predicts nothing — found via a 10M-visit turn
    that put 18% on a Psychic into a Dark-type (immune). ~5% of consults;
    split out so the gate is judged on turns where prediction can matter."""
    tops = []
    for w in rec.get("worlds") or []:
        rows = w.get("rows") or []
        if rows:
            tops.append(max(q for _, _, q in rows))
    if not tops:
        return False
    return min(tops) >= 0.95 or max(tops) <= 0.05


def opp_bucket(name: str) -> str:
    if name == "richwoman":
        return "richwoman"
    if name.startswith("LLM-"):
        return "llm-bots"
    return "other"


def band(turn: int) -> str:
    return ("T1-3" if turn <= 3 else "T4-9" if turn <= 9
            else "T10-25" if turn <= 25 else "T26+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default="showdown/overlay_shadow.jsonl")
    ap.add_argument("--logs",
                    default="showdown/bench/overnight_*_ladder.log")
    ap.add_argument("--name", default="PAC-Crystal")
    ap.add_argument("--matched-only", action="store_true",
                    help="restrict to turns with a gemma prediction so "
                         "engine/uniform are scored on the same population")
    args = ap.parse_args()

    decisions, opp_name = parse_logs(sorted(glob.glob(args.logs)),
                                     args.name)
    print(f"log decisions parsed: {len(decisions)} opponent turns "
          f"across {len(opp_name)} rooms")

    policies = ("gemma", "engine", "uniform")
    tot = Counter()          # (policy, metric) -> value
    n_all = n_pred = 0
    outside = 0
    h2h = Counter()          # gemma-vs-engine head-to-head on disagreements
    split = defaultdict(Counter)   # (dim, key) -> counters

    for line in open(args.shadow):
        d = json.loads(line)
        if args.matched_only and not d.get("reply_pred"):
            continue
        actual = decisions.get((d["tag"], d["turn"]))
        if not actual:
            continue
        options = {collapse(o) for o in (d.get("reply_options") or
                                         engine_dist(d).keys())}
        if len(options) < 2:
            continue
        e_dist = engine_dist(d)
        if not e_dist:
            continue
        u_dist = {o: 1.0 / len(options) for o in sorted(options)}
        g_raw = d.get("reply_pred")
        g_dist = None
        if g_raw:
            g_dist = defaultdict(float)
            for k, v in g_raw.items():
                g_dist[collapse(k)] += v
        n_all += 1
        outside += (actual not in options)
        ob = opp_bucket(opp_name.get(d["tag"], "?"))
        tb = band(d["turn"])
        st = "decided" if decided(d) else "contested"
        for pol, dist in (("engine", e_dist), ("uniform", u_dist),
                          ("gemma", g_dist)):
            if dist is None:
                continue
            if pol == "gemma":
                n_pred += 1
            ranked = sorted(dist.items(), key=lambda kv: -kv[1])
            top1 = ranked[0][0] if ranked else None
            hit1 = (top1 == actual)
            hit3 = actual in {o for o, _ in ranked[:3]}
            tot[(pol, "n")] += 1
            tot[(pol, "top1")] += hit1
            tot[(pol, "top3")] += hit3
            tot[(pol, "brier")] += brier(dist, actual, options)
            split[("opp", ob)][(pol, "n")] += 1
            split[("opp", ob)][(pol, "top1")] += hit1
            split[("band", tb)][(pol, "n")] += 1
            split[("band", tb)][(pol, "top1")] += hit1
            split[("state", st)][(pol, "n")] += 1
            split[("state", st)][(pol, "top1")] += hit1
        if g_dist:
            g_top = max(g_dist.items(), key=lambda kv: kv[1])[0]
            e_top = max(e_dist.items(), key=lambda kv: kv[1])[0]
            if g_top != e_top:
                h2h["n"] += 1
                h2h["gemma"] += (g_top == actual)
                h2h["engine"] += (e_top == actual)

    print(f"joined scoreable turns: {n_all} "
          f"(with a gemma prediction: {n_pred})")
    print(f"actual outside searched options: {outside}/{n_all} = "
          f"{outside / max(n_all, 1):.1%}  <- the 'killers' blind spot")
    print(f"\n{'policy':>8} {'n':>6} {'top-1':>7} {'top-3':>7} "
          f"{'brier':>7}")
    for pol in policies:
        n = tot[(pol, "n")]
        if not n:
            print(f"{pol:>8} {0:>6}      —       —       —")
            continue
        print(f"{pol:>8} {n:>6} {tot[(pol, 'top1')] / n:>7.1%} "
              f"{tot[(pol, 'top3')] / n:>7.1%} "
              f"{tot[(pol, 'brier')] / n:>7.3f}")
    if h2h["n"]:
        print(f"\nhead-to-head (gemma top-1 != engine top-1, "
              f"n={h2h['n']}): gemma right {h2h['gemma']}, "
              f"engine right {h2h['engine']}")
        m = h2h["gemma"] + h2h["engine"]
        if m:
            z = (h2h["gemma"] - m / 2) / (math.sqrt(m) / 2)
            p = math.erfc(abs(z) / math.sqrt(2))
            print(f"sign test on {m} decisive: z={z:+.2f}, "
                  f"two-sided p={p:.3f}")
    for dim, title in (("opp", "per-opponent top-1"),
                       ("band", "per-turn-band top-1"),
                       ("state", "contested-vs-decided top-1")):
        print(f"\n{title}:")
        for (d2, key), c in sorted(split.items()):
            if d2 != dim:
                continue
            row = f"  {key:12s}"
            for pol in policies:
                n = c[(pol, "n")]
                row += (f" {pol}={c[(pol, 'top1')] / n:.1%}(n={n})"
                        if n else f" {pol}=—")
            print(row)


if __name__ == "__main__":
    main()
