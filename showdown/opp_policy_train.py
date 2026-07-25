#!/usr/bin/env python3
"""Opponent-policy net: featurize streamed policy corpora and train.

Predicts SIDE TWO's next action from a state — the inference orientation
(our sampled worlds hold the opponent as side_two). Every record yields up
to two samples: (state, s2_action) and (side-flipped state, s1_action) —
the flip is a two-segment swap in the state string, the same augmentation
policy_train.py and bench_value_net.py already use. Action space is 14
slot classes for side_two: [m0..m3, m0-tera..m3-tera, party0..5]. Records
whose key can't be resolved against the engine-parsed state (set-
reconstruction divergence: 8.4% human / 4.2% fp at extraction) drop here —
this is the --drop-illegal pass.

featurize: policy pkl -> disk memmaps (X f32 [N,2738], y u8, mask u16
bitfield, gid i64). RAM stays flat regardless of corpus size; the 16.9GB
human corpus streams. --side-prefix keeps only samples whose ACTING player
name starts with the prefix (FPSpar1 selects foul-play's decisions from
the bench corpus, both seats, via the flip).

train: by-GAME split (gid hash — the by-position leak faked 0.999 once),
per-batch gather from memmaps (the value-net OOM lesson), masked
cross-entropy, and CALIBRATION as a first-class metric: top-1/top-3, NLL,
10-bin ECE, and high-confidence reliability (accuracy + coverage at
p>=0.9 / p>=0.95) — the point is knowing whether "95% Earthquake" means 95%.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

N_CLASSES = 14


def _gid_hash(game_id: str) -> int:
    return int.from_bytes(hashlib.sha1(game_id.encode()).digest()[:8],
                          "big") % (2 ** 62)


def _flip(state_str: str) -> str:
    major = state_str.split("/")
    return "/".join([major[1], major[0]] + major[2:])


def _resolve(state, key: str):
    """(label, legality_mask) for side_two of an engine state."""
    side = state.side_two
    active_idx = int(side.active_index)
    active = side.pokemon[active_idx]
    mask = 0
    can_tera = not any(p.terastallized for p in side.pokemon)
    move_ids = []
    for i, mv in enumerate(active.moves):
        if i >= 4:
            break
        mid = str(mv.id).lower()
        move_ids.append(mid)
        if mv.pp > 0 and not mv.disabled and mid != "none":
            mask |= 1 << i
            if can_tera:
                mask |= 1 << (i + 4)
    for i, p in enumerate(side.pokemon):
        if p.hp > 0 and i != active_idx:
            mask |= 1 << (i + 8)

    label = None
    k = key.lower()
    if k.startswith("switch "):
        target = k[7:]
        for i, p in enumerate(side.pokemon):
            if str(p.id).lower() == target:
                label = i + 8
                break
    else:
        tera = k.endswith("-tera")
        base = k[:-5] if tera else k
        for i, mid in enumerate(move_ids):
            if mid == base:
                label = i + (4 if tera else 0)
                break
    if label is not None and not (mask >> label) & 1:
        label = None  # recorded action illegal in this state
    return label, mask


def _samples(rec, meta, side_prefix):
    """Yield (state_str_oriented, key) for each usable side of a record."""
    p1 = str((meta or {}).get("p1") or "")
    p2 = str((meta or {}).get("p2") or "")
    if rec["s2"] and (not side_prefix or p2.startswith(side_prefix)):
        yield rec["state"], rec["s2"]
    if rec["s1"] and (not side_prefix or p1.startswith(side_prefix)):
        yield _flip(rec["state"]), rec["s1"]


def cmd_featurize(args) -> int:
    import poke_engine as pe
    from showdown.featurizer_v3 import parse_state_v3, STATE_V3_FEATURES
    from showdown.replay_to_policy_gen9 import load_policy_games

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    games_seen = 0
    for _w, gid, meta, recs in load_policy_games(args.corpus):
        games_seen += 1
        n += sum(1 for r in recs for _ in _samples(r, meta, args.side_prefix))
        if args.limit_games and games_seen >= args.limit_games:
            break
    print(f"pass1: {games_seen} games, {n} candidate samples", flush=True)

    X = np.lib.format.open_memmap(f"{out}.X.npy", mode="w+",
                                  dtype=np.float32,
                                  shape=(n, STATE_V3_FEATURES))
    y = np.lib.format.open_memmap(f"{out}.y.npy", mode="w+",
                                  dtype=np.uint8, shape=(n,))
    msk = np.lib.format.open_memmap(f"{out}.mask.npy", mode="w+",
                                    dtype=np.uint16, shape=(n,))
    gids = np.lib.format.open_memmap(f"{out}.gid.npy", mode="w+",
                                     dtype=np.int64, shape=(n,))

    w = dropped = parse_err = 0
    games_seen = 0
    t0 = time.time()
    for _win, gid, meta, recs in load_policy_games(args.corpus):
        games_seen += 1
        gh = _gid_hash(gid)
        for r in recs:
            for state_str, key in _samples(r, meta, args.side_prefix):
                try:
                    st = pe.State.from_string(state_str)
                    label, mask = _resolve(st, key)
                except Exception:
                    parse_err += 1
                    continue
                if label is None or mask == 0:
                    dropped += 1
                    continue
                X[w] = parse_state_v3(state_str)
                y[w] = label
                msk[w] = mask
                gids[w] = gh
                w += 1
        if games_seen % 5000 == 0:
            print(f"  {games_seen} games, {w} written, {dropped} dropped, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if args.limit_games and games_seen >= args.limit_games:
            break

    np.save(f"{out}.n.npy", np.array([w], dtype=np.int64))
    print(f"featurize done in {time.time() - t0:.0f}s: {w} samples "
          f"({dropped} illegal-dropped, {parse_err} parse errors) -> "
          f"{out}.*.npy")
    return 0


def _ece(probs: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs > lo) & (probs <= hi)
        if sel.sum():
            ece += sel.mean() * abs(correct[sel].mean() - probs[sel].mean())
    return float(ece)


def cmd_train(args) -> int:
    import torch
    import torch.nn as nn

    n = int(np.load(f"{args.data}.n.npy")[0])
    X = np.load(f"{args.data}.X.npy", mmap_mode="r")
    y = np.load(f"{args.data}.y.npy", mmap_mode="r")
    msk = np.load(f"{args.data}.mask.npy", mmap_mode="r")
    gids = np.load(f"{args.data}.gid.npy", mmap_mode="r")
    dim = X.shape[1]

    val_sel = (gids[:n] % args.val_mod) == 0
    tr_idx = np.where(~val_sel)[0]
    va_idx = np.where(val_sel)[0]
    tr_idx = tr_idx[tr_idx < n]
    va_idx = va_idx[va_idx < n]

    # RAM-resident working set: the X memmap (30.7GB for 100k human games)
    # exceeds RAM, so per-batch disk gather starves the GPU (measured 1%
    # util, 32min/epoch). Subsample train rows to --ram-cap and hold X as an
    # fp16 RAM tensor; gather is then a fast in-memory index. --ram-cap 0
    # keeps the old disk-memmap path. Subsampling is by ROW but the val set
    # stays whole and by-GAME (never subsampled), so the split is honest.
    ram = args.ram_cap > 0
    if ram and len(tr_idx) > args.ram_cap:
        tr_idx = np.random.default_rng(0).choice(tr_idx, args.ram_cap,
                                                 replace=False)
    print(f"{n} samples, dim {dim}; by-game split "
          f"{len(tr_idx)} train / {len(va_idx)} val"
          + (f" (RAM-resident fp16)" if ram else ""))

    dev = torch.device(args.device)
    widths = [int(w) for w in args.hidden.split(",") if w]
    layers = []
    prev = dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        if args.dropout > 0:
            layers.append(nn.Dropout(args.dropout))
        prev = w
    layers.append(nn.Linear(prev, N_CLASSES))
    model = nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bit = torch.tensor([1 << i for i in range(N_CLASSES)], dtype=torch.int32)
    print(f"model {[dim] + widths + [N_CLASSES]} dropout={args.dropout} "
          f"wd={args.wd} cosine-lr", flush=True)

    ram_store = {}
    if ram:
        t_load = time.time()
        for tag, idx in (("tr", tr_idx), ("va", va_idx)):
            sidx = np.sort(idx)
            # chunked fp16 load: never materialize the full fp32 gather (a
            # 2.7M-row fp32 spike would blow past RAM); cast per 200k chunk
            xr = torch.empty((len(sidx), dim), dtype=torch.float16)
            for c in range(0, len(sidx), 200_000):
                blk = sidx[c:c + 200_000]
                xr[c:c + len(blk)] = torch.from_numpy(
                    np.ascontiguousarray(X[blk])).half()
            ram_store[tag] = (
                sidx, xr,
                torch.from_numpy(y[sidx].astype(np.int64)),
                torch.from_numpy(msk[sidx].astype(np.int32)),
            )
        gb = sum(v[1].nbytes for v in ram_store.values()) / 1073741824
        print(f"loaded {gb:.1f}GB fp16 into RAM in {time.time()-t_load:.0f}s",
              flush=True)

    def batches(idx, bs, shuffle, tag=None):
        if ram and tag is not None:
            sidx, Xr, yr, mr = ram_store[tag]
            pos = np.random.permutation(len(sidx)) if shuffle \
                else np.arange(len(sidx))
            for i in range(0, len(pos), bs):
                p = pos[i:i + bs]
                xb = Xr[p].float().to(dev)
                yb = yr[p].to(dev)
                legal = ((mr[p].unsqueeze(1) & bit.unsqueeze(0)) != 0).to(dev)
                yield xb, yb, legal
            return
        order = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(order), bs):
            sel = np.sort(order[i:i + bs])  # sorted memmap gather is faster
            xb = torch.from_numpy(np.ascontiguousarray(X[sel])).to(dev)
            yb = torch.from_numpy(y[sel].astype(np.int64)).to(dev)
            mb = torch.from_numpy(msk[sel].astype(np.int32))
            legal = ((mb.unsqueeze(1) & bit.unsqueeze(0)) != 0).to(dev)
            yield xb, yb, legal

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for xb, yb, legal in batches(tr_idx, args.batch, shuffle=True, tag="tr"):
            logits = model(xb).masked_fill(~legal, float("-inf"))
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(yb)
            seen += len(yb)
        model.eval()
        top1 = top3 = nll = 0.0
        probs_all, corr_all = [], []
        with torch.no_grad():
            for xb, yb, legal in batches(va_idx, args.batch, shuffle=False, tag="va"):
                logits = model(xb).masked_fill(~legal, float("-inf"))
                p = torch.softmax(logits, dim=1)
                nll += float(nn.functional.cross_entropy(
                    logits, yb, reduction="sum"))
                top = p.argmax(1)
                top1 += float((top == yb).sum())
                top3 += float((p.topk(3, dim=1).indices == yb.unsqueeze(1))
                              .any(1).sum())
                probs_all.append(p.max(1).values.cpu().numpy())
                corr_all.append((top == yb).cpu().numpy())
        nv = max(1, len(va_idx))
        probs = np.concatenate(probs_all)
        corr = np.concatenate(corr_all).astype(np.float64)
        line = (f"epoch {epoch}: train_loss={tot / max(1, seen):.4f} "
                f"val top1={top1 / nv:.3f} top3={top3 / nv:.3f} "
                f"nll={nll / nv:.4f} ece={_ece(probs, corr):.4f}")
        for thr in (0.9, 0.95):
            sel = probs >= thr
            acc = corr[sel].mean() if sel.any() else float("nan")
            line += f" | p>={thr}: acc={acc:.3f} cov={sel.mean():.4f}"
        print(line + f" lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)
        sched.step()
        torch.save(model.state_dict(), args.model_out)

    print(f"saved {args.model_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("featurize")
    f.add_argument("--corpus", required=True)
    f.add_argument("--out-prefix", required=True)
    f.add_argument("--limit-games", type=int, default=0)
    f.add_argument("--side-prefix", default=None,
                   help="keep only samples whose acting player's name "
                        "starts with this (FPSpar1 = foul-play's decisions)")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="featurize out-prefix")
    t.add_argument("--model-out", default="showdown/opp_policy_v1.pt")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--device", default="cuda")
    t.add_argument("--val-mod", type=int, default=20)
    t.add_argument("--ram-cap", type=int, default=0,
                   help="hold up to N train rows as an fp16 RAM tensor "
                        "(fast gather); 0 = disk memmap. ~13GB per 2.4M rows")
    t.add_argument("--hidden", default="512,256",
                   help="comma-separated trunk widths")
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--wd", type=float, default=0.0)
    args = ap.parse_args()
    return cmd_featurize(args) if args.cmd == "featurize" else cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
