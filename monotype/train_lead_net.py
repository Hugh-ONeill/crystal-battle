#!/usr/bin/env python3
"""
Train the LeadPickerNet on supervised replay data.

Each replay gives 2 training examples by symmetry: (P1 view -> P1 lead)
and (P2 view -> P2 lead). 80/20 train/val split, Adam + cross-entropy,
~20 epochs is enough on this dataset size.

Usage:
  .venv/bin/python monotype/train_lead_net.py \\
      --data monotype/lead_train_data.npz \\
      --out monotype/lead_net.pt --epochs 30
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

from monotype.lead_net import LeadPickerNet


def load_symmetric_dataset(path: Path) -> TensorDataset:
    """Load npz and build a (2N, 6, MON_DIM) own / opp / label dataset.

    The first N rows are P1's view (own=P1, opp=P2, label=P1 lead).
    The next N are P2's view (own=P2, opp=P1, label=P2 lead).
    """
    d = np.load(path)
    X_p1, X_p2 = d["X_p1"], d["X_p2"]
    y_p1, y_p2 = d["y_p1"], d["y_p2"]
    own = np.concatenate([X_p1, X_p2], axis=0)
    opp = np.concatenate([X_p2, X_p1], axis=0)
    labels = np.concatenate([y_p1, y_p2], axis=0)
    return TensorDataset(
        torch.from_numpy(own),
        torch.from_numpy(opp),
        torch.from_numpy(labels),
    )


def evaluate(model, loader, device) -> tuple[float, float, float]:
    """Return (mean_loss, top1_acc, top3_acc)."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    n = 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for own, opp, y in loader:
            own, opp, y = own.to(device), opp.to(device), y.to(device)
            logits = model(own, opp)
            total_loss += crit(logits, y).item()
            top1 = logits.argmax(dim=-1)
            total_top1 += (top1 == y).sum().item()
            top3 = logits.topk(3, dim=-1).indices  # (B, 3)
            total_top3 += (top3 == y.unsqueeze(-1)).any(dim=-1).sum().item()
            n += y.size(0)
    return total_loss / n, total_top1 / n, total_top3 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== device: {device} ===")

    dataset = load_symmetric_dataset(args.data)
    n = len(dataset)
    n_val = int(n * args.val_frac)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"=== dataset: {n} examples ({n_train} train / {n_val} val) ===")

    model = LeadPickerNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== model: {n_params} params ===")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for own, opp, y in train_loader:
            own, opp, y = own.to(device), opp.to(device), y.to(device)
            logits = model(own, opp)
            loss = crit(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        train_loss /= n_seen
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, device)
        print(f"  epoch {epoch:3d}  train_loss {train_loss:.3f}  "
              f"val_loss {val_loss:.3f}  val_top1 {val_top1*100:5.1f}%  "
              f"val_top3 {val_top3*100:5.1f}%")
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), args.out)

    print(f"\n=== best val top-1 = {best_val_top1*100:.1f}% (saved {args.out}) ===")
    print("baselines: random 16.7%, majority-class ~21%")


if __name__ == "__main__":
    main()
