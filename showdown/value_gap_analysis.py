#!/usr/bin/env python3
"""Value-net campaign, STEP 0 — quantify the gaps (no training, decisive).

Tests the premise the whole value-net-in-the-gaps line rests on: does the STATIC
eval predict game outcomes WORSE in the regimes we think it fails (late-game /
long stall) than in normal play? If static eval predicts outcomes about equally
well everywhere, the premise is wrong and the line dies here — before a GPU hour.

Signal: static-eval win-prob = sigmoid(evaluate(state)/150) (value_train.py's own
mapping) vs the actual winner, stratified by total alive mons (late-game proxy)
and by game length (stall proxy). Metrics per stratum: AUC (discrimination),
accuracy, log-loss (calibration+discrimination), base rate.

Corpus: showdown/gen9ou_policy_human.pkl (winner 1=side_one, 2=side_two, 0=draw).
"""
import math
import sys
from collections import defaultdict

import poke_engine as pe

sys.path.insert(0, ".")
from showdown.replay_to_policy_gen9 import load_policy_games

CORPUS = "showdown/gen9ou_policy_human.pkl"
SCALE = 150.0  # RESIDUAL_EVAL_SCALE, matches value_train.py
MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def auc(pairs):
    """Mann-Whitney AUC: P(pred_pos > pred_neg). pairs = [(prob, label01)]."""
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    # rank-sum
    allv = sorted(((p, 1) for p in pos), key=lambda t: t[0])
    merged = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    rank = 0.0
    i = 0
    ranksum_pos = 0.0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average rank for ties (1-indexed)
        for k in range(i, j):
            if merged[k][1] == 1:
                ranksum_pos += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (ranksum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def report(name, rows):
    if len(rows) < 30:
        print(f"  {name:22s} n={len(rows):5d}  (too few)")
        return
    n = len(rows)
    base = sum(y for _, y in rows) / n
    acc = sum(1 for p, y in rows if (p >= 0.5) == (y == 1)) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in rows) / n
    a = auc(rows)
    print(f"  {name:22s} n={n:5d}  AUC={a:.3f}  acc={acc:.3f}  "
          f"logloss={ll:.3f}  base(s1win)={base:.2f}")


def main():
    by_alive = defaultdict(list)
    by_len = defaultdict(list)
    overall = []
    games = 0
    for winner, gid, meta, recs in load_policy_games(CORPUS):
        if games >= MAX_GAMES:
            break
        games += 1
        if winner not in (1, 2):
            continue
        y = 1 if winner == 1 else 0          # side_one won?
        glen = len(recs)
        lenbucket = "short(<25)" if glen < 25 else "mid(25-49)" if glen < 50 else "long(>=50)"
        for k, rec in enumerate(recs):
            if k % 2:                        # subsample every other record
                continue
            try:
                st = pe.State.from_string(rec["state"])
                ev = pe.evaluate(st)
                alive = sum(1 for p in st.side_one.pokemon if p.hp > 0) + \
                        sum(1 for p in st.side_two.pokemon if p.hp > 0)
            except Exception:
                continue
            p = sigmoid(ev / SCALE)
            overall.append((p, y))
            ab = "early(>=10)" if alive >= 10 else "mid(6-9)" if alive >= 6 else "late(<=5)"
            by_alive[ab].append((p, y))
            by_len[lenbucket].append((p, y))

    print(f"\n=== STEP 0: static-eval outcome-prediction, {games} games, "
          f"{len(overall)} states ===")
    report("OVERALL", overall)
    print("\n-- by alive-mon count (late-game = the gap) --")
    for k in ("early(>=10)", "mid(6-9)", "late(<=5)"):
        report(k, by_alive[k])
    print("\n-- by game length (long = stall proxy) --")
    for k in ("short(<25)", "mid(25-49)", "long(>=50)"):
        report(k, by_len[k])
    print("\nREAD: if AUC/acc DROP sharply in late(<=5) and long(>=50), the gaps are")
    print("real and a learned value has room there. If flat, the premise is weak.")


if __name__ == "__main__":
    main()
