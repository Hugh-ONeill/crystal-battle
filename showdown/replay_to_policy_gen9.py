#!/usr/bin/env python3
"""Extract opponent-policy training data from gen9 replays.

Emits (state, action) supervision for the opponent-policy net: for every
decision in every replay, the PRE-decision engine state plus the action each
side actually took, keyed in the _norm_opt space the MCTS result strings use
("moveid", "moveid-tera", "switch species") so the net's output aligns with
s2 priors with no translation layer.

Output pickle:
    [(winner_int, game_id, {p1,p2,ratings}, [{"state","s1","s2","kind"}, ...]), ...]
Game identity is preserved for BY-GAME train/val splits — the by-position
leak invented a 0.999 accuracy for the value net; never again.

Corpora: --replays dir of Showdown replay .json, or --jsonl (metamon dump /
bench_logs_to_replays.py output). Same flags as replay_to_training_gen9.

--verify N samples N random records, rebuilds each state, and checks the
recorded action against root_get_all_options: the legality rate is the
dataset's alignment score with the engine's option space (divergence from
imperfect replay stepping shows up here — know the number before training).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

from showdown.chaos_stats import ChaosStats
from showdown.replay_to_trajectory import replay_to_trajectory


def _label_to_winner(label: float) -> int:
    return {1.0: 1, 0.0: 2}.get(label, 0)


def verify_sample(games, n, seed=0):
    """Legality-check: rebuild each sampled state and test the recorded key
    against the engine's own root option strings (a 1ms search enumerates
    them in exactly the comparable to_string space). Reports the alignment
    rate — imperfect replay stepping shows up here, so know the number
    before training on the data."""
    import poke_engine as pe
    rng = random.Random(seed)
    records = [rec for _w, _gid, _m, recs in games for rec in recs]
    if not records:
        return
    picks = rng.sample(records, min(n, len(records)))
    ok = bad = unbuildable = 0
    mism = []
    for rec in picks:
        try:
            st = pe.State.from_string(rec["state"])
            res = pe.monte_carlo_tree_search(st, 1)
        except Exception:
            unbuildable += 1
            continue
        for key, rows in ((rec["s1"], res.side_one), (rec["s2"], res.side_two)):
            if key is None:
                continue
            opts = {r.move_choice.lower() for r in rows}
            if key.lower() in opts:
                ok += 1
            else:
                bad += 1
                if len(mism) < 5:
                    mism.append((key, sorted(opts)[:6]))
    total = ok + bad
    if total:
        print(f"verify: {ok}/{total} recorded actions legal in rebuilt state "
              f"({100 * ok / total:.1f}%), {unbuildable} states unbuildable")
        for key, opts in mism:
            print(f"  mismatch: {key!r} not in {opts}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=str, default="showdown/replays/gen9ou")
    ap.add_argument("--jsonl", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default="showdown/gen9ou_policy_data.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-records", type=int, default=4,
                    help="Drop games with fewer than N decision records")
    ap.add_argument("--verify", type=int, default=0,
                    help="Legality-check N random records after extraction")
    args = ap.parse_args()

    random.seed(args.seed)
    chaos = ChaosStats(format="gen9ou")

    def _iter_replays():
        if args.jsonl:
            with open(args.jsonl) as fh:
                for ln, line in enumerate(fh):
                    if args.limit and ln >= args.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield f"line{ln}", json.loads(line)
                    except Exception:
                        continue
        else:
            files = sorted(Path(args.replays).glob("*.json"))
            if args.limit > 0:
                files = files[: args.limit]
            for f in files:
                try:
                    yield f.name, json.load(open(f))
                except Exception:
                    continue

    src = args.jsonl or args.replays
    print(f"extracting policy records from {src} (limit={args.limit or 'all'})")

    games = []
    n_turn = n_pivot = n_short = n_err = n_empty = 0
    t0 = time.time()
    for i, (name, data) in enumerate(_iter_replays()):
        try:
            traj, acts = replay_to_trajectory(data, chaos, record_actions=True)
        except Exception:
            n_err += 1
            continue
        if not acts:
            n_empty += 1
            continue
        if len(acts) < args.min_records:
            n_short += 1
            continue
        winner = _label_to_winner(traj[0][1]) if traj else 0
        # player names let the trainer pick a specific side's actions
        # (fp-corpus: which side is FPSpar1*) and rating-filter humans
        from showdown.replay_parse_gen9 import parse_replay
        meta = {}
        try:
            pt = parse_replay(data)
            meta = {"p1": pt.p1_name, "p2": pt.p2_name,
                    "p1_rating": pt.p1_rating, "p2_rating": pt.p2_rating}
        except Exception:
            pass
        games.append((winner, name, meta, acts))
        n_turn += sum(1 for a in acts if a["kind"] == "turn")
        n_pivot += sum(1 for a in acts if a["kind"] == "pivot")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} replays, {len(games)} kept, "
                  f"{n_turn + n_pivot} records, {time.time() - t0:.0f}s")

    labels = sum(1 for _w, _g, _m, recs in games for r in recs
                 for k in (r["s1"], r["s2"]) if k)
    print(f"done in {time.time() - t0:.0f}s: {len(games)} games, "
          f"{n_turn} turn + {n_pivot} pivot records, {labels} action labels "
          f"(skipped: {n_short} short, {n_empty} empty, {n_err} err)")
    with open(args.out, "wb") as fh:
        pickle.dump(games, fh)
    print(f"wrote {args.out}")

    if args.verify:
        verify_sample(games, args.verify, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
