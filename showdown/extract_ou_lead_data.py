#!/usr/bin/env python3
"""Build supervised lead-pick training data from gen9ou replays.

WHY (2026-08-07). `_lead_ev_blend` blends the preview maximin matrix with EV
under the OPPONENT'S observed lead distribution — but it sources that
distribution from scouting_book.json, so it fires for exactly the handful of
opponents we have already played and NEVER for an unseen October entrant,
where we silently fall back to plain maximin. A net that predicts a lead from
BOTH PREVIEWS ALONE generalises that blend to any opponent.

This is the OPPONENT-PREDICTION product, deliberately: the campaign's record
is that learned components convert when they predict the opponent and fail
when they pilot us (the move-net reached 55% top-1 and REGRESSED the bench as
a pilot, after which its standing verdict became "opponent prediction only").
Target here is therefore "which mon did this side lead", trained on ALL games
regardless of who won. The our-own-lead variant is a different product with a
different estimator — see the TODO; do not conflate them.

Differences from the monotype extractor this is modelled on:
  - rosters come from |poke| lines, same as monotype, but pastes are built
    from the PS curated set index (showdown/ps_sets.py) instead of
    type-keyed canonical sets, since OU has no team type to key on;
  - the featurizer (monotype/featurizer_lead_preview.py) is already
    format-agnostic — dual typing, role flags, stat block — so it is reused
    unchanged. The monotype-specific type embedding lives in the MOVE net,
    not this one.

Output matches the monotype npz contract so train_lead_net.py works as-is:
  X_p1, X_p2 (N, 6, 51) | y_p1, y_p2 (N,) | winner (N,)

Usage:
  .venv/bin/python showdown/extract_ou_lead_data.py \
      --replays-dir showdown/replays/gen9ou --out showdown/ou_lead_data.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from monotype.featurizer_lead_preview import featurize_preview  # noqa: E402
from showdown.ps_sets import get_index  # noqa: E402

# CHAOS FALLBACK. The curated PS index carries 118 species — the OU core —
# but replays contain the whole tier, so requiring a curated set for all 12
# mons dropped 75% of replays on a 60-replay probe (51 distinct species
# missing). Chaos covers 408 species, so it backfills the tail. Sets built
# this way are modal-move/modal-item guesses, which is exactly the right
# fidelity for a PREVIEW-time model that cannot know the real set anyway.
_CHAOS = None


def _chaos_index() -> dict:
    global _CHAOS
    if _CHAOS is None:
        raw = json.loads((HERE / "gen9ou_chaos.json").read_text())
        _CHAOS = {_norm(k): v for k, v in (raw.get("data") or raw).items()}
    return _CHAOS


def _chaos_block(species: str) -> str | None:
    v = _chaos_index().get(_norm(species))
    if not v:
        return None
    top = lambda d, n=1: [k for k, _ in sorted(
        (d or {}).items(), key=lambda kv: -kv[1])[:n]]
    moves = top(v.get("Moves"), 4)
    if not moves:
        return None
    item = (top(v.get("Items")) or ["Leftovers"])[0]
    ability = (top(v.get("Abilities")) or ["No Ability"])[0]
    spread = (top(v.get("Spreads")) or ["Serious:0/0/0/0/0/0"])[0]
    nature, _, evs = spread.partition(":")
    names = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
    ev_str = " / ".join(f"{n} {lab}" for n, lab in zip(evs.split("/"), names)
                        if n.isdigit() and int(n))
    lines = [f"{species} @ {item}", f"Ability: {ability}"]
    if ev_str:
        lines.append(f"EVs: {ev_str}")
    lines.append(f"{nature or 'Serious'} Nature")
    lines += [f"- {m}" for m in moves]
    return "\n".join(lines)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def parse_replay(data: dict) -> dict | None:
    """roster + leads + winner from one replay JSON. Leads are the FIRST
    switch on each side, which precedes |turn|1."""
    log = data.get("log", "")
    if not log:
        return None
    team = {"p1": [], "p2": []}
    lead = {"p1": None, "p2": None}
    name = {"p1": None, "p2": None}
    winner = None
    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        p = line.split("|")
        if len(p) < 2:
            continue
        cmd = p[1]
        if cmd == "player" and len(p) >= 4:
            if p[2] in name:
                name[p[2]] = p[3]
        elif cmd == "poke" and len(p) >= 4:
            if p[2] in team:
                team[p[2]].append(p[3].split(",")[0].strip())
        elif cmd == "switch" and len(p) >= 4:
            side = p[2][:2]
            if side in lead and lead[side] is None:
                lead[side] = p[3].split(",")[0].strip()
        elif cmd == "win" and len(p) >= 3:
            winner = p[2].strip()
    if len(team["p1"]) != 6 or len(team["p2"]) != 6:
        return None
    if not lead["p1"] or not lead["p2"] or not winner:
        return None
    return {"team": team, "lead": lead, "name": name, "winner": winner}


def roster_to_paste(roster: list[str], index) -> str | None:
    """6-mon paste from the PS curated index. The heaviest-weighted set is
    used: this is a PREVIEW-time model, so we cannot know the real set and
    the modal one is the honest prior. Any species without a set kills the
    replay rather than silently substituting a wrong statline."""
    blocks = []
    for sp in roster:
        cands = index.candidates.get(_norm(sp))
        if not cands:
            base = _norm(sp.split("-")[0])
            cands = index.candidates.get(base)
        if not cands:
            blk = _chaos_block(sp)          # tail species -> chaos modal set
            if not blk:
                return None
            blocks.append(blk)
            continue
        s = max(cands, key=lambda c: c.get("weight", 0))
        evs = s.get("evs") or {}
        ev_str = " / ".join(f"{v} {k.capitalize()}"
                            for k, v in evs.items() if v)
        lines = [f"{sp} @ {s.get('item', 'Leftovers')}",
                 f"Ability: {s.get('ability', 'No Ability')}"]
        if s.get("tera_type"):
            lines.append(f"Tera Type: {s['tera_type']}")
        if ev_str:
            lines.append(f"EVs: {ev_str}")
        lines.append(f"{s.get('nature', 'Serious')} Nature")
        lines += [f"- {m}" for m in (s.get("moves") or [])[:4]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("showdown/replays/gen9ou"))
    ap.add_argument("--out", type=Path, default=Path("showdown/ou_lead_data.npz"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    index = get_index("gen9ou")
    files = sorted(args.replays_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    print(f"=== {len(files)} replays in {args.replays_dir} ===")

    X1, X2, Y1, Y2, W = [], [], [], [], []
    skipped = {"parse": 0, "paste": 0, "lead_not_in_roster": 0}
    for i, f in enumerate(files):
        if i and i % 500 == 0:
            print(f"  {i}/{len(files)}  kept={len(X1)}  skipped={skipped}")
        try:
            rec = parse_replay(json.loads(f.read_text()))
        except Exception:
            rec = None
        if not rec:
            skipped["parse"] += 1
            continue
        p1 = roster_to_paste(rec["team"]["p1"], index)
        p2 = roster_to_paste(rec["team"]["p2"], index)
        if not p1 or not p2:
            skipped["paste"] += 1
            continue
        try:
            i1 = [_norm(s) for s in rec["team"]["p1"]].index(
                _norm(rec["lead"]["p1"]))
            i2 = [_norm(s) for s in rec["team"]["p2"]].index(
                _norm(rec["lead"]["p2"]))
        except ValueError:
            skipped["lead_not_in_roster"] += 1
            continue
        try:
            f1, f2 = featurize_preview(p1, p2)
        except Exception:
            skipped["paste"] += 1
            continue
        X1.append(f1)
        X2.append(f2)
        Y1.append(i1)
        Y2.append(i2)
        W.append(1 if rec["winner"] == rec["name"]["p1"] else 0)

    print(f"\nkept {len(X1)} replays | skipped {skipped}")
    if not X1:
        sys.exit("no usable replays")
    np.savez_compressed(
        args.out,
        X_p1=np.asarray(X1, dtype=np.float32),
        X_p2=np.asarray(X2, dtype=np.float32),
        y_p1=np.asarray(Y1, dtype=np.int64),
        y_p2=np.asarray(Y2, dtype=np.int64),
        winner=np.asarray(W, dtype=np.int64))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
