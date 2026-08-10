#!/usr/bin/env python3
"""Extract per-game surrogate-endpoint records from bench lane logs.

The NEXT UP surrogate work (TODO top-of-file): winrate is one bit per game
and verification cost scales with the inverse square of the effect, so we
need denser endpoints. This parses the banked `<corpus>_L*_ours.log` files
into one JSON record per game carrying every candidate endpoint the logs can
support:

  - winner_us / tie / forfeit / turns
  - terminal margin (alive_us - alive_them at |win|) and terminal HP sums
  - state snapshots at the start of turns T in --checkpoints (default
    8,12,16): per-side HP-fraction sum (unrevealed mons count 1.0), alive
    count, hazard layers, status count among living mons

Validation design (preregistered in TODO): candidates are judged against the
HISTORICAL A/B library — known nulls must stay null, known effects must
resolve at smaller n — not by in-sample correlation with the win label.

Usage:
  .venv/bin/python showdown/surrogate_extract.py dittoab3_0806 k1commit ...
      [--bench-dir showdown/bench] [--out-dir showdown/bench/surrogate]
      [--checkpoints 8,12,16]

Output: <out-dir>/<corpus>.jsonl, one file per corpus (cache-friendly:
re-running a corpus overwrites only its own file).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GAME_HDR = re.compile(r"^=== lane (\d+) game (\d+)/(\d+) team: (\S+) ")
HP_RE = re.compile(r"^(\d+)(?:/(\d+))?")

HAZARDS = {"stealthrock": "rocks", "spikes": "spikes",
           "toxicspikes": "tspikes", "stickyweb": "web"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class Game:
    def __init__(self, lane, no, total, pairing, checkpoints):
        self.meta = {"lane": lane, "game": no, "total": total,
                     "pairing": pairing}
        self.checkpoints = checkpoints
        self.sides = {}                  # p1/p2 -> player name
        self.our_side = None
        self.hp = defaultdict(dict)      # side -> {nick: frac}
        self.fainted = defaultdict(set)  # side -> {nick}
        self.status = defaultdict(set)   # side -> {nick}
        self.haz = defaultdict(lambda: defaultdict(int))  # side->{h:layers}
        self.turn = 0
        self.cp = {}
        self.winner = None
        self.tie = False
        self.forfeit = False

    def side_state(self, side):
        hp = self.hp[side]
        return {
            "hp": round(sum(hp.values()) + (6 - len(hp)), 3),
            "alive": 6 - len(self.fainted[side]),
            "status": len(self.status[side] - self.fainted[side]),
            "haz": dict(self.haz[side]),
        }

    def snapshot(self, turn):
        if turn in self.checkpoints and self.our_side:
            them = "p2" if self.our_side == "p1" else "p1"
            self.cp[str(turn)] = {"us": self.side_state(self.our_side),
                                  "them": self.side_state(them)}

    def record(self, corpus, our_name):
        if not self.our_side:
            for s, n in self.sides.items():
                if n == our_name:
                    self.our_side = s
        if not self.our_side:
            return None
        them = "p2" if self.our_side == "p1" else "p1"
        us_st, them_st = self.side_state(self.our_side), self.side_state(them)
        winner_us = None
        if self.winner is not None:
            winner_us = (self.winner == self.sides.get(self.our_side))
        return {
            "corpus": corpus, **self.meta,
            "our_side": self.our_side,
            "winner_us": winner_us, "tie": self.tie,
            "forfeit": self.forfeit, "turns": self.turn,
            "margin": us_st["alive"] - them_st["alive"],
            "hp_margin": round(us_st["hp"] - them_st["hp"], 3),
            "alive_us": us_st["alive"], "alive_them": them_st["alive"],
            "cp": self.cp,
        }


def parse_hp(field, game, side, nick):
    m = HP_RE.match(field)
    if not m:
        return
    if field.startswith("0"):
        game.hp[side][nick] = 0.0
        game.fainted[side].add(nick)
        return
    cur = int(m.group(1))
    mx = int(m.group(2)) if m.group(2) else 100
    game.hp[side][nick] = cur / max(mx, 1)
    game.fainted[side].discard(nick)      # Revival Blessing etc.


def parse_file(path, corpus, checkpoints, out):
    our_name = None
    game = None
    n_flushed = 0
    for raw in open(path, errors="replace"):
        m = GAME_HDR.match(raw)
        if m:
            if game is not None:
                rec = game.record(corpus, our_name)
                if rec:
                    out.write(json.dumps(rec) + "\n")
                    n_flushed += 1
            game = Game(int(m.group(1)), int(m.group(2)),
                        int(m.group(3)), m.group(4), checkpoints)
            continue
        if game is None:
            continue
        line = ANSI.sub("", raw.rstrip("\n"))
        if our_name is None:
            m = re.search(r" - (\S+) - INFO - ", line)
            if m:
                our_name = m.group(1)
        # protocol payload: raw continuation lines start with '|';
        # inline server lines carry '<<< |...'
        if line.startswith("|"):
            payload = line
        else:
            i = line.find("<<< ")
            if i < 0:
                continue
            payload = line[i + 4:]
            if not payload.startswith("|"):
                continue
        p = payload.split("|")
        cmd = p[1] if len(p) > 1 else ""
        if cmd == "player" and len(p) > 3:
            if p[2] in ("p1", "p2"):
                game.sides[p[2]] = p[3]
                if p[3] == our_name:
                    game.our_side = p[2]
        elif cmd == "turn":
            game.turn = int(p[2])
            game.snapshot(game.turn)
        elif cmd in ("switch", "drag", "replace") and len(p) > 4:
            side, nick = p[2][:2], p[2].split(": ", 1)[-1]
            parse_hp(p[4], game, side, nick)
        elif cmd in ("-damage", "-heal", "-sethp") and len(p) > 3:
            side, nick = p[2][:2], p[2].split(": ", 1)[-1]
            parse_hp(p[3], game, side, nick)
        elif cmd == "faint" and len(p) > 2:
            side, nick = p[2][:2], p[2].split(": ", 1)[-1]
            game.fainted[side].add(nick)
            game.hp[side][nick] = 0.0
        elif cmd == "-status" and len(p) > 3:
            side, nick = p[2][:2], p[2].split(": ", 1)[-1]
            game.status[side].add(nick)
        elif cmd == "-curestatus" and len(p) > 2:
            side, nick = p[2][:2], p[2].split(": ", 1)[-1]
            game.status[side].discard(nick)
        elif cmd == "-sidestart" and len(p) > 3:
            h = HAZARDS.get(_norm(p[3].replace("move: ", "")))
            if h:
                game.haz[p[2][:2]][h] += 1
        elif cmd == "-sideend" and len(p) > 3:
            h = HAZARDS.get(_norm(p[3].replace("move: ", "")))
            if h:
                game.haz[p[2][:2]].pop(h, None)
        elif cmd == "win" and len(p) > 2:
            game.winner = p[2].strip()
        elif cmd == "tie":
            game.tie = True
        elif "forfeited." in payload:
            game.forfeit = True
    if game is not None:
        rec = game.record(corpus, our_name)
        if rec:
            out.write(json.dumps(rec) + "\n")
            n_flushed += 1
    return n_flushed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--bench-dir", type=Path, default=Path("showdown/bench"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("showdown/bench/surrogate"))
    ap.add_argument("--checkpoints", default="8,12,16")
    args = ap.parse_args()

    cps = {int(t) for t in args.checkpoints.split(",")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for corpus in args.corpora:
        files = sorted(glob.glob(str(args.bench_dir /
                                     f"{corpus}_L*_ours.log")))
        if not files:
            print(f"{corpus}: NO LANE LOGS, skipped")
            continue
        n = 0
        with open(args.out_dir / f"{corpus}.jsonl", "w") as out:
            for f in files:
                n += parse_file(f, corpus, cps, out)
        print(f"{corpus}: {n} games from {len(files)} lanes")


if __name__ == "__main__":
    main()
