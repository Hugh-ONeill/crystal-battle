#!/usr/bin/env python3
"""Offline calibration gate for the opponent-policy net.

The net only earns a live wiring if it beats the priors the search can
ALREADY use. On held-out policy records, score three predictors of side
two's next action:

  uniform : flat over legal options (the floor)
  chaos   : Smogon usage move-frequencies for the active species, restricted
            to its legal moves and renormalized (the informed classical
            baseline the search consumes today; move-decisions only — chaos
            cannot predict switches)
  net     : the trained opponent-policy net

Two comparisons, because chaos cannot play on switch turns:
  MOVE decisions  : net vs chaos vs uniform (does it beat usage stats?)
  ALL decisions   : net vs uniform (the total value, incl. switch prediction)

Metrics per predictor: top-1, top-3, mean NLL (log-loss on the taken
action), and 10-bin ECE on the top-1 probability. NLL is the number that
matters for priors — it rewards putting mass on the right action and
punishes false confidence, exactly the failure mode (team-archive) we are
guarding against.

Holdout must be games the net never trained on:
  --holdout skip:N   use games after the first N (human net trained on
                     --limit-games 100000 -> skip:100000 is unseen)
  --holdout valmod:M keep games with gid_hash %% M == 0 (fp net trained on
                     all games; this is its by-game val partition)
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from showdown.opp_policy_train import N_CLASSES, _gid_hash, _flip, _resolve

_MOVE_SLOTS = set(range(8))     # m0..m3, m0-tera..m3-tera
_SWITCH_SLOTS = set(range(8, 14))


def _load_net(path):
    import torch
    import torch.nn as nn
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        widths, dim, drop = ckpt["hidden"], ckpt["dim"], ckpt.get("dropout", 0)
        sd = ckpt["state_dict"]
    else:                       # bare state_dict fallback (legacy)
        sd = ckpt
        dim = sd["0.weight"].shape[1]
        widths = []
        i = 0
        while f"{i}.weight" in sd:
            w = sd[f"{i}.weight"].shape[0]
            if w != N_CLASSES:
                widths.append(w)
            i += 2
        drop = 0
    layers = []
    prev = dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        if drop > 0:
            layers.append(nn.Dropout(drop))
        prev = w
    layers.append(nn.Linear(prev, N_CLASSES))
    model = nn.Sequential(*layers)
    model.load_state_dict(sd)
    model.eval()
    return model


class Scorer:
    def __init__(self, name):
        self.name = name
        self.n = 0
        self.top1 = 0
        self.top3 = 0
        self.nll = 0.0
        self.pbin = np.zeros(10)     # summed top-prob per bin
        self.cbin = np.zeros(10)     # summed correct per bin
        self.nbin = np.zeros(10)

    def add(self, dist: np.ndarray, label: int):
        # dist: prob over 14 classes, already legal-masked and normalized
        order = np.argsort(-dist)
        self.n += 1
        self.top1 += int(order[0] == label)
        self.top3 += int(label in order[:3])
        p = max(dist[label], 1e-9)
        self.nll += -math.log(p)
        top_p = dist[order[0]]
        b = min(9, int(top_p * 10))
        self.pbin[b] += top_p
        self.cbin[b] += int(order[0] == label)
        self.nbin[b] += 1

    def report(self) -> str:
        if not self.n:
            return f"{self.name:8s}  (no records)"
        ece = 0.0
        for b in range(10):
            if self.nbin[b]:
                ece += (self.nbin[b] / self.n) * abs(
                    self.cbin[b] / self.nbin[b] - self.pbin[b] / self.nbin[b])
        return (f"{self.name:8s}  top1={self.top1 / self.n:.3f} "
                f"top3={self.top3 / self.n:.3f} nll={self.nll / self.n:.4f} "
                f"ece={ece:.4f}  (n={self.n})")


def _legal_slots(mask: int) -> list[int]:
    return [i for i in range(N_CLASSES) if (mask >> i) & 1]


def _uniform_dist(mask: int) -> np.ndarray:
    d = np.zeros(N_CLASSES)
    legal = _legal_slots(mask)
    for i in legal:
        d[i] = 1.0 / len(legal)
    return d


def _chaos_dist(state, chaos, mask) -> np.ndarray | None:
    """Usage-frequency distribution over the active mon's legal MOVE slots.
    None if the species isn't in chaos. Tera slots share their base move's
    mass split evenly (chaos has no per-turn tera signal)."""
    side = state.side_two
    active = side.pokemon[int(side.active_index)]
    species = str(active.id).lower()
    mon = chaos.pokemon.get(species)
    if mon is None:
        return None
    mp = mon.move_probs()
    d = np.zeros(N_CLASSES)
    for i in range(4):
        if not (mask >> i) & 1 and not (mask >> (i + 4)) & 1:
            continue
        mid = str(active.moves[i].id).lower()
        w = mp.get(mid, 0.0)
        if (mask >> i) & 1:
            d[i] += w
        if (mask >> (i + 4)) & 1:      # tera variant legal too
            d[i + 4] += w
    s = d.sum()
    if s <= 0:                          # no usage overlap -> uniform on moves
        return None
    return d / s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--holdout", required=True,
                    help="skip:N or valmod:M")
    ap.add_argument("--limit-games", type=int, default=20000)
    ap.add_argument("--side-prefix", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from showdown.featurizer_v3 import parse_state_v3
    from showdown.chaos_stats import ChaosStats
    from showdown.replay_to_policy_gen9 import load_policy_games
    import poke_engine as pe

    mode, val = args.holdout.split(":")
    val = int(val)
    chaos = ChaosStats(format="gen9ou")
    net = _load_net(args.net).to(args.device)

    def _held(gid, i):
        if mode == "skip":
            return i >= val
        return _gid_hash(gid) % val == 0

    # move-decision scorers (all three predictors) + all-decision (net/uniform)
    m_uni, m_cha, m_net = Scorer("uniform"), Scorer("chaos"), Scorer("net")
    a_uni, a_net = Scorer("uniform*"), Scorer("net*")

    seen = kept = 0
    t0 = time.time()
    batch_states, batch_meta = [], []

    def flush():
        if not batch_states:
            return
        X = np.stack([parse_state_v3(s) for s in batch_states])
        with torch.no_grad():
            logits = net(torch.from_numpy(X).float().to(args.device))
        for row, (label, mask, st) in zip(logits.cpu().numpy(), batch_meta):
            legal = np.array([(mask >> i) & 1 for i in range(N_CLASSES)],
                             dtype=bool)
            z = np.where(legal, row, -1e9)
            z = z - z.max()
            e = np.exp(z) * legal
            ndist = e / e.sum()
            a_uni.add(_uniform_dist(mask), label)
            a_net.add(ndist, label)
            if label in _MOVE_SLOTS:
                cd = _chaos_dist(st, chaos, mask)
                m_uni.add(_uniform_dist(mask), label)
                m_net.add(ndist, label)
                m_cha.add(cd if cd is not None else _uniform_dist(mask), label)
        batch_states.clear()
        batch_meta.clear()

    for _w, gid, meta, recs in load_policy_games(args.corpus):
        seen += 1
        if not _held(gid, seen - 1):
            continue
        kept += 1
        for r in recs:
            for state_str, key in (((r["state"], r["s2"]),
                                    (_flip(r["state"]), r["s1"]))):
                if not key:
                    continue
                if args.side_prefix:
                    who = meta.get("p2" if state_str == r["state"] else "p1", "")
                    if not str(who).startswith(args.side_prefix):
                        continue
                try:
                    st = pe.State.from_string(state_str)
                    label, mask = _resolve(st, key)
                except Exception:
                    continue
                if label is None or mask == 0:
                    continue
                batch_states.append(state_str)
                batch_meta.append((label, mask, st))
        if len(batch_states) >= 4096:
            flush()
        if kept >= args.limit_games:
            break
    flush()

    print(f"held-out {kept} games, {m_net.n} move + "
          f"{a_net.n - m_net.n} switch decisions, {time.time() - t0:.0f}s\n")
    print("MOVE decisions (net must beat chaos to be worth wiring):")
    for s in (m_uni, m_cha, m_net):
        print("  " + s.report())
    print("\nALL decisions (net vs uniform; includes switch prediction):")
    for s in (a_uni, a_net):
        print("  " + s.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
