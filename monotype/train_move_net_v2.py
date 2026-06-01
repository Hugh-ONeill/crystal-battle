#!/usr/bin/env python3
"""
Train MoveNetV2 — same as v1 but with the 53-dim dynamic state context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from monotype.move_net import MoveNetV2


def load_dataset(path: Path) -> tuple[TensorDataset, int, list[str]]:
    d = np.load(path, allow_pickle=True)
    vocab = list(d["move_vocab"])
    n_moves = len(vocab)
    cand = d["candidate_moves"].astype(np.int64)
    cand[cand < 0] = n_moves
    ds = TensorDataset(
        torch.from_numpy(d["actor_features"]),
        torch.from_numpy(d["opp_features"]),
        torch.from_numpy(d["state_features"]),
        torch.from_numpy(cand),
        torch.from_numpy(d["hp_actor"]),
        torch.from_numpy(d["hp_opp"]),
        torch.from_numpy(d["team_type_actor"]),
        torch.from_numpy(d["team_type_opp"]),
        torch.from_numpy(d["y"]),
    )
    return ds, n_moves, vocab


def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, top1, top2, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            actor, opp, ts, cand, hpa, hpo, tta, tto, y = batch
            logits = model(actor, opp, ts, cand, hpa, hpo, tta, tto)
            total_loss += crit(logits, y).item()
            top1 += (logits.argmax(dim=-1) == y).sum().item()
            top2 += (logits.topk(2, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1).sum().item()
            n += y.size(0)
    return total_loss / n, top1 / n, top2 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== device: {device} ===")

    ds, n_moves, vocab = load_dataset(args.data)
    n = len(ds)
    n_val = int(n * args.val_frac)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"=== dataset: {n} examples ({n-n_val} train / {n_val} val), vocab {n_moves} ===")

    model = MoveNetV2(n_moves=n_moves).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== model: {n_params} params ===")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_seen = 0.0, 0
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            actor, opp, ts, cand, hpa, hpo, tta, tto, y = batch
            logits = model(actor, opp, ts, cand, hpa, hpo, tta, tto)
            loss = crit(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        train_loss /= n_seen
        vl, t1, t2 = evaluate(model, val_loader, device)
        print(f"  epoch {epoch:3d}  train_loss {train_loss:.3f}  "
              f"val_loss {vl:.3f}  val_top1 {t1*100:5.1f}%  val_top2 {t2*100:5.1f}%")
        if t1 > best:
            best = t1
            torch.save({"state_dict": model.state_dict(),
                        "vocab": vocab, "n_moves": n_moves}, args.out)

    print(f"\n=== best val top-1 = {best*100:.1f}% (saved {args.out}) ===")
    print("baselines: random 25%, majority-class (canonical-top) 37%, V1 net 52.3%")


if __name__ == "__main__":
    main()
