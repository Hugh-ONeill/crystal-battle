#!/usr/bin/env python3
"""
Walk a trace_match.py output file and flag engine antipatterns.

Antipatterns counted:
  - support_at_low_hp: chose a setup/screen/hazard/recovery move when active
    HP fraction was < 30% and the move doesn't deal direct damage.
  - tied_visits: top-3 visit counts within 1.3x of the max -> no clear read.
  - switch_off_kill: switched out (move starts with "switch ") when the
    opposing active was at < 25% HP -> walked off a likely KO.
  - low_branch_repeat: same move picked 3+ turns in a row AND the sampled
    branch's `p` stayed below 40% (flinch / miss / para loop).

For each (P1, P2) side, prints counts and the turn list where each fired.

Usage:
  .venv/bin/python monotype/analyze_trace.py monotype/traces/*.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Self-attached support: only benefits the user mon, so wasted if the
# mon dies the same/next turn. THIS is the real antipattern at low HP.
SELF_BUFF_MOVES = {
    "swordsdance", "nastyplot", "calmmind", "bulkup", "irondefense",
    "amnesia", "agility", "rockpolish", "shellsmash", "shiftgear",
    "dragondance", "quiverdance", "noretreat", "geomancy",
    "cosmicpower", "stockpile", "acidarmor", "barrier", "defendorder",
}
# Team-persistent support: screens, hazards, weather, terrain, status on
# opponent. CORRECT at low HP for suicide-lead setters (Klefki, Ninetales-A,
# Grimmsnarl, Tornadus, etc.) — do NOT flag as antipattern.
TEAM_PERSISTENT_MOVES = {
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
    "reflect", "lightscreen", "auroraveil",
    "trickroom", "tailwind", "magicroom", "wonderroom", "gravity",
    "memento", "healingwish", "lunardance", "partingshot",
    "willowisp", "thunderwave", "toxic", "spore", "sleeppowder", "yawn",
    "stunspore", "glare", "hypnosis", "darkvoid",
    "taunt", "encore", "destinybond",
    "sunnyday", "raindance", "snowscape", "chillyreception", "sandstorm",
    "electricterrain", "psychicterrain", "grassyterrain", "mistyterrain",
    "rapidspin", "defog", "courtchange",
}
# Recovery moves: correct at low HP if they heal meaningfully.
RECOVERY_MOVES = {
    "recover", "roost", "slackoff", "softboiled", "moonlight",
    "morningsun", "synthesis", "wish", "painsplit", "shoreup",
    "milkdrink", "rest", "lifedew",
    "substitute",  # arguable, but often used as defensive setup
    "protect", "detect", "kingsshield", "spikyshield", "banefulbunker",
    "silktrap", "burningbulwark",
}

TURN_RE = re.compile(r"^\[T(\d+)\]")
ACTIVE_RE = re.compile(
    r"^\s+(P[12])=([A-Z\-]+)\((\d+)/(\d+)[\),]"
)
PICK_RE = re.compile(
    r"^\s+(P[12]) picks: (.+?)\s{2,}\[(.+)\]\s*$"
)
BRANCH_RE = re.compile(r"applied branch p=([\d.]+)%")
RESULT_RE = re.compile(r"\[T(\d+)\] (P[12]) wins")


def parse_visits(visit_str: str) -> list[tuple[str, int]]:
    """Parse 'move1(N1), move2(N2), ...' into [(move, n), ...]."""
    out = []
    for chunk in visit_str.split(", "):
        m = re.match(r"(.+?)\((\d+)\)$", chunk.strip())
        if m:
            out.append((m.group(1).strip(), int(m.group(2))))
    return out


def analyze(path: Path) -> dict:
    lines = path.read_text().splitlines()
    # per-turn state we accumulate before flushing on each branch line
    turn_no = 0
    active = {"P1": None, "P2": None}   # (id, hp, maxhp)
    picks = {"P1": None, "P2": None}    # (move, visits_list)
    last_pick = {"P1": [], "P2": []}    # rolling history of (move, branch_p)
    branch_p_pending = None

    flags = {
        "P1": defaultdict(list),
        "P2": defaultdict(list),
    }
    result = None
    result_turn = None

    def flush_turn():
        nonlocal branch_p_pending
        if turn_no == 0:
            return
        # Apply flags based on what we just saw this turn.
        for side in ("P1", "P2"):
            opp = "P2" if side == "P1" else "P1"
            mine = active[side]
            other = active[opp]
            pick = picks[side]
            if not mine or not other or not pick:
                continue
            move, visits = pick
            id_, hp, mx = mine
            if mx <= 0:
                continue
            hp_frac = hp / mx
            opp_hp_frac = other[1] / other[2] if other[2] > 0 else 1.0
            mv_norm = move.replace(" ", "").lower()

            # self_buff_at_low_hp: self-attached boost on a dying mon.
            # Team-persistent setup and recovery are CORRECT at low HP
            # (suicide-lead screen setters etc.) so they're excluded.
            if hp_frac < 0.30 and mv_norm in SELF_BUFF_MOVES:
                flags[side]["self_buff_at_low_hp"].append(
                    f"T{turn_no} {id_}({hp}/{mx}) -> {move}"
                )

            # tied_visits (top-3 within 1.3x)
            if len(visits) >= 3:
                vs = sorted(v for _, v in visits)[-3:]
                if vs[0] >= vs[-1] / 1.3 and vs[-1] > 1000:
                    top3 = sorted(visits, key=lambda x: -x[1])[:3]
                    flags[side]["tied_visits"].append(
                        f"T{turn_no} {id_} top3: "
                        + ", ".join(f"{m}({v})" for m, v in top3)
                    )

            # switch_off_kill: own active still healthy (>50%) AND opp at
            # <25% HP AND we chose to switch. This filters sacrifice-switches
            # (where own active dies next turn anyway).
            if (move.startswith("switch ")
                    and opp_hp_frac < 0.25
                    and hp_frac > 0.50):
                flags[side]["switch_off_kill"].append(
                    f"T{turn_no} {id_}({hp}/{mx}) -> {move}  "
                    f"(opp {other[0]} at {other[1]}/{other[2]})"
                )

            # low_branch_repeat (track per side)
            last_pick[side].append((mv_norm, branch_p_pending))
            last_pick[side] = last_pick[side][-3:]
            if (len(last_pick[side]) == 3
                    and len({m for m, _ in last_pick[side]}) == 1
                    and all((p is not None and p < 40) for _, p in last_pick[side])):
                flags[side]["low_branch_repeat"].append(
                    f"T{turn_no} {id_} stuck on '{mv_norm}' "
                    + "/".join(f"{p:.0f}%" for _, p in last_pick[side])
                )

    for ln in lines:
        m = TURN_RE.match(ln)
        if m:
            # New turn header: this turn's picks/active will follow; the
            # branch line of the *previous* turn has already been logged.
            flush_turn()
            turn_no = int(m.group(1))
            active = {"P1": None, "P2": None}
            picks = {"P1": None, "P2": None}
            branch_p_pending = None
            # detect "P2 wins"
            mr = RESULT_RE.search(ln)
            if mr:
                result_turn = int(mr.group(1))
                result = mr.group(2)
            continue

        m = ACTIVE_RE.match(ln)
        if m:
            side, mon_id, hp, mx = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            active[side] = (mon_id, hp, mx)
            continue

        m = PICK_RE.match(ln)
        if m:
            side, move, vlist = m.group(1), m.group(2), m.group(3)
            picks[side] = (move, parse_visits(vlist))
            continue

        m = BRANCH_RE.search(ln)
        if m:
            branch_p_pending = float(m.group(1))
            continue

    # final flush
    flush_turn()

    return {
        "path": str(path),
        "result": result,
        "result_turn": result_turn,
        "flags": {side: dict(d) for side, d in flags.items()},
    }


def fmt_report(reports: list[dict], verbose: bool = False) -> str:
    out = []
    out.append(f"\n=== {'file':<50}  result  turns  P1{'flags':>17}    P2 flags")
    for r in reports:
        f1 = r["flags"]["P1"]; f2 = r["flags"]["P2"]
        def short(d):
            return " ".join(f"{k}:{len(v)}" for k, v in d.items() if v) or "-"
        name = Path(r["path"]).name
        result = r["result"] or "?"
        turns = r["result_turn"] or "?"
        out.append(f"  {name:<50}  {result}     {turns!s:>4}    P1[{short(f1)}]  P2[{short(f2)}]")

    if verbose:
        for r in reports:
            out.append(f"\n--- {Path(r['path']).name} ({r['result']} wins on T{r['result_turn']}) ---")
            for side in ("P1", "P2"):
                d = r["flags"][side]
                if not any(d.values()):
                    continue
                out.append(f"  {side}:")
                for cat, hits in d.items():
                    if not hits:
                        continue
                    out.append(f"    {cat} ({len(hits)}):")
                    for h in hits:
                        out.append(f"      {h}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+", type=Path)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    reports = [analyze(t) for t in args.traces]
    reports.sort(key=lambda r: r["path"])
    print(fmt_report(reports, verbose=args.verbose))


if __name__ == "__main__":
    main()
