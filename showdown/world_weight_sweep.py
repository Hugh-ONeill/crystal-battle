#!/usr/bin/env python3
"""Offline coupling sweep for the CB_WORLD_WEIGHTS dial over the overlay
shadow corpus (overlay_shadow.jsonl).

Why (branch-invariance law): the 2026-08-06 shadow read found the curated
world matches future set reveals 55.7/44.3 over the hedge world, which makes
"upweight world 0" a learning-free candidate. But the merge docstring is
explicit that the hedge world's job is robustness, not realism — so before
any paired A/B we measure OFFLINE how hard the weight dial couples to
decisions at all, and against the oracle ceiling (perfect per-record world
knowledge scored from what the game later revealed). If even the oracle
barely moves decisions, no picker — static, LLM, or perfect — can matter,
and the A/B is not worth its bench-nights.

Reconstruction: the engine's default merge is stake-weighted VISIT SHARES
(_merge_mcts_results, non-raw path). We reproduce it from each record's
logged top-8 rows; the tail truncation makes this approximate, so the sweep
validates itself first by recomputing the shadow's own λ-blend flips from
the logged rows and reporting agreement with the live-computed ones.

Flip rates are measured on the CONSULTED-TURN distribution (gated turns:
opening / near-tie / world-disagreement / role-rules) — exactly the turns
where the dial could matter; quiet turns dilute nothing here.

Usage:
  .venv/bin/python showdown/world_weight_sweep.py [--shadow PATH]
      [--grid 0.55,0.6,0.65,0.7,0.8,0.9,1.0]
"""

import argparse
import json
from collections import Counter, defaultdict

LAMBDAS = ("0.25", "0.5", "1.0")


def top_action(worlds, weights):
    """Stake-weighted visit-share merge over logged top-8 rows -> top action.
    Mirrors _merge_mcts_results' default path (share = visits/total per
    world, stake-weighted sum across worlds)."""
    agg = defaultdict(float)
    for wi, w in enumerate(worlds):
        rows = w.get("rows") or []
        total = sum(r[1] for r in rows)
        if total <= 0:
            continue
        stake = weights[wi] if wi < len(weights) else 0.0
        for move, visits, _q in rows:
            agg[move] += stake * (visits / total)
    if not agg:
        return None
    return max(agg.items(), key=lambda kv: kv[1])[0]


def kind(a, b):
    sa, sb = a.startswith("switch"), b.startswith("switch")
    return ("sw->sw" if sa and sb else "mv->sw" if sb
            else "sw->mv" if sa else "mv->mv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default="showdown/overlay_shadow.jsonl")
    ap.add_argument("--grid", default="0.55,0.6,0.65,0.7,0.8,0.9,1.0")
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",")]

    recs = []
    final = defaultdict(lambda: defaultdict(
        lambda: {"moves": set(), "item": None}))
    for line in open(args.shadow):
        d = json.loads(line)
        recs.append(d)
        for mon, info in (d.get("appendix", {}).get("their_mons")
                          or {}).items():
            e = final[d["tag"]][mon]
            e["moves"].update(info.get("revealed_moves") or [])
            if info.get("item"):
                e["item"] = info["item"]

    # ---- usable records + self-validation against the live-computed flips
    usable = []
    val_n = val_ok = 0
    for d in recs:
        worlds = d.get("worlds") or []
        if len(worlds) != 2:
            continue
        base = top_action(worlds, [1.0, 1.0])
        if base is None:
            continue
        usable.append((d, base))
        f = d.get("flips")
        if isinstance(f, dict) and f.get("llm_weights"):
            lw = f["llm_weights"]
            for lam in LAMBDAS:
                logged = f.get(lam, {}).get("top")
                if logged is None:
                    continue
                lam_f = float(lam)
                blend = [lam_f * w + (1 - lam_f) * 0.5 for w in lw]
                mine = top_action(worlds, blend)
                val_n += 1
                val_ok += (mine == logged)

    print(f"usable K=2 records: {len(usable)}/{len(recs)}")
    print(f"reconstruction check vs live-computed λ-blend tops: "
          f"{val_ok}/{val_n} = {val_ok / max(val_n, 1):.1%}")

    # ---- oracle adjudication per record (future reveals decide the world)
    decidable = []          # (d, base, winner_world)
    for d, base in usable:
        known = {m: set(i.get("revealed_moves") or []) for m, i in
                 (d.get("appendix", {}).get("their_mons") or {}).items()}
        known_items = {m: i.get("item") for m, i in
                       (d.get("appendix", {}).get("their_mons")
                        or {}).items()}
        scores = [0, 0]
        for wi in (0, 1):
            assumed = d["worlds"][wi].get("assumed_sets") or {}
            for mon, fin in final[d["tag"]].items():
                a = assumed.get(mon)
                if not a:
                    continue
                fut = fin["moves"] - known.get(mon, set())
                scores[wi] += sum(1 for mv in fut
                                  if mv in (a.get("moves") or []))
                if (fin["item"] and not known_items.get(mon)
                        and fin["item"] == a.get("item")):
                    scores[wi] += 1
        if scores[0] != scores[1]:
            decidable.append((d, base, 0 if scores[0] > scores[1] else 1))

    print(f"oracle-decidable records: {len(decidable)}  "
          f"(world 0 truer in "
          f"{sum(1 for _, _, w in decidable if w == 0)})")

    # ---- oracle ceiling
    o_flip = 0
    o_agree_base = 0
    oracle_top = {}
    for d, base, winner in decidable:
        ot = top_action(d["worlds"], [1.0, 0.0] if winner == 0
                        else [0.0, 1.0])
        oracle_top[id(d)] = ot
        if ot != base:
            o_flip += 1
        else:
            o_agree_base += 1
    n_dec = len(decidable)
    print(f"\nORACLE CEILING (hard-collapse to the truer world, decidable "
          f"records):")
    print(f"  flips vs equal-vote baseline: {o_flip}/{n_dec} = "
          f"{o_flip / max(n_dec, 1):.1%}")

    # ---- static sweep
    print(f"\nSTATIC SWEEP (weights [w0, 1-w0], all usable records; "
          f"margin classes from the engine's own live margin):")
    header = (f"  {'w0':>5} {'flips':>11} {'tie<=.03':>9} {'ovr>=.10':>9} "
              f"{'mv->sw':>7} {'sw->mv':>7} {'oracle-agree':>13}")
    print(header)
    base_oracle_agree = sum(
        1 for d, base, _ in decidable if oracle_top[id(d)] == base)
    for w0 in [0.5] + grid:
        flips = 0
        ties = ovr = 0
        kinds = Counter()
        agree = 0
        for d, base in usable:
            t = top_action(d["worlds"], [w0, 1.0 - w0])
            if t != base:
                flips += 1
                m = d.get("engine_margin", 1.0)
                ties += (m <= 0.03)
                ovr += (m >= 0.10)
                kinds[kind(base, t)] += 1
        for d, base, _ in decidable:
            t = top_action(d["worlds"], [w0, 1.0 - w0])
            if t == oracle_top[id(d)]:
                agree += 1
        n = len(usable)
        print(f"  {w0:>5.2f} {flips:>5}={flips / n:>5.1%} {ties:>9} "
              f"{ovr:>9} {kinds['mv->sw']:>7} {kinds['sw->mv']:>7} "
              f"{agree:>6}/{n_dec} = {agree / max(n_dec, 1):.1%}")
    print(f"  (baseline w0=0.50 oracle-agreement: "
          f"{base_oracle_agree}/{n_dec} = "
          f"{base_oracle_agree / max(n_dec, 1):.1%})")

    # ---- asymmetry decomposition: agreement gains when the CURATED world
    # was real are mild (took the aggressive line correctly); losses when
    # the HEDGE world was real are the dangerous class (ignored a real
    # Scarf-shaped threat). Report the exchange rate per setting.
    print("\nDECOMPOSITION (oracle-agreement delta vs baseline, by which "
          "world was real):")
    print(f"  {'w0':>5} {'truth=curated':>15} {'truth=hedge':>13} "
          f"{'net':>6}")
    for w0 in grid:
        d_cur = d_hedge = 0
        for d, base, winner in decidable:
            t = top_action(d["worlds"], [w0, 1.0 - w0])
            ot = oracle_top[id(d)]
            was = (base == ot)
            now = (t == ot)
            if was == now:
                continue
            delta = 1 if now else -1
            if winner == 0:
                d_cur += delta
            else:
                d_hedge += delta
        print(f"  {w0:>5.2f} {d_cur:>+15d} {d_hedge:>+13d} "
              f"{d_cur + d_hedge:>+6d}")


if __name__ == "__main__":
    main()
