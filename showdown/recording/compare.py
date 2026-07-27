#!/usr/bin/env python3
"""Beat-by-beat comparison: what ACTUALLY happened in the game (GROUND TRUTH,
parsed from the gen9 player's raw Showdown protocol log) interleaved with what
the duo SAID (the commentary transcript from capture_feed). Surfaces
hallucinations (a beat claims something the protocol never shows) and drops (a
real event no beat mentioned) — it caught every mirror-attribution flip during
the duo's debugging.

Run:  .venv/bin/python -m showdown.recording.compare \
          <player_log> <transcript> <out.txt> [--username CBDemo]

The player log is poke-env at --log-level 20: the first protocol message of
each received frame is prefixed '... - <user> - INFO - <<< ', later messages in
the frame are bare '|verb|...' lines; sent choices are '>>> ' and skipped.
Commentary tagged [Tn] recaps the exchange going INTO turn n, so it lines up
with the GAME actions of turn n-1..n (noted in the output header).
"""
from __future__ import annotations

import argparse
import re

_PREFIX = re.compile(r"^.* - \S+ - (?:INFO|DEBUG) - (.*)$")
_STAT = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD",
         "spe": "Spe", "accuracy": "acc", "evasion": "eva"}


def protocol_messages(path: str):
    """Yield each received protocol message as a '|'-split list, in order."""
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        m = _PREFIX.match(line)
        if m:
            line = m.group(1)
            if line.startswith("<<< "):
                line = line[4:]
            elif line.startswith(">>> "):
                continue          # a choice we sent, not a game event
            # else: a plain log line (search output) — kept only if protocol
        line = line.strip()
        if not line.startswith("|"):
            continue              # room ids, search logs, blanks
        yield line.split("|")     # ['', 'verb', arg, ...]


def _sp(species_by_pos: dict, token: str) -> str:
    """'p2a: Nickname' -> tracked species (nickname-proof)."""
    pos = token.split(":")[0]
    return species_by_pos.get(pos, token.split(": ", 1)[-1])


def build_game(path: str, username: str):
    """turn -> list of plain-English action strings. Turn 0 = leads/setup.
    Returns (turns, winner, role_us) where role_us is 'p1'/'p2' for us."""
    role_us = None
    species_by_pos: dict[str, str] = {}
    turns: dict[int, list] = {0: []}
    cur = 0
    result = None

    def side(pos: str) -> str:
        return "us" if pos[:2] == role_us else "them"

    def add(s: str):
        turns.setdefault(cur, []).append(s)

    for m in protocol_messages(path):
        if len(m) < 2:
            continue
        v = m[1]
        if v == "player" and len(m) > 3:
            if m[3] == username:
                role_us = m[2]
        elif v == "turn":
            cur = int(m[2])
            turns.setdefault(cur, [])
        elif v in ("switch", "drag") and len(m) > 3:
            pos = m[2].split(":")[0]
            species = m[3].split(",")[0]
            species_by_pos[pos] = species
            hp = m[4].split(" ")[0] if len(m) > 4 else "?"
            add(f"{side(m[2])}: sent in {species} ({hp})")
        elif v == "move" and len(m) > 3:
            user = _sp(species_by_pos, m[2])
            tgt = _sp(species_by_pos, m[4]) if len(m) > 4 else ""
            add(f"{side(m[2])}: {user} used {m[3]}"
                + (f" -> {tgt}" if tgt else ""))
        elif v == "-crit":
            add("    (critical hit)")
        elif v == "-supereffective":
            add("    (super effective)")
        elif v == "-resisted":
            add("    (resisted)")
        elif v == "-immune":
            add("    (no effect / immune)")
        elif v == "-miss":
            add("    (missed)")
        elif v == "-damage" and len(m) > 3:
            hp = m[3].split(" ")[0]
            frm = next((a.split(": ")[-1][len("[from] "):]
                        if a.startswith("[from] ") else None
                        for a in m[4:] if a.startswith("[from] ")), None)
            add(f"    {_sp(species_by_pos, m[2])} -> {hp}"
                + (f"  (from {frm})" if frm else ""))
        elif v == "-start" and len(m) > 3:
            add(f"    {_sp(species_by_pos, m[2])} gained "
                f"{m[3].split(': ')[-1]}")
        elif v == "-activate" and len(m) > 2 and "confusion" in "".join(m[3:]):
            add(f"    {_sp(species_by_pos, m[2])} is confused")
        elif v == "-heal" and len(m) > 3:
            hp = m[3].split(" ")[0]
            add(f"    {_sp(species_by_pos, m[2])} healed -> {hp}")
        elif v == "faint" and len(m) > 2:
            add(f"  ** {_sp(species_by_pos, m[2])} FAINTED ({side(m[2])})")
        elif v == "-status" and len(m) > 3:
            add(f"    {_sp(species_by_pos, m[2])} was {m[3]}")
        elif v == "-curestatus" and len(m) > 3:
            add(f"    {_sp(species_by_pos, m[2])} cured of {m[3]}")
        elif v == "-terastallize" and len(m) > 3:
            add(f"  >> {_sp(species_by_pos, m[2])} Terastallized to {m[3]}")
        elif v in ("-boost", "-unboost") and len(m) > 4:
            arrow = "+" if v == "-boost" else "-"
            add(f"    {_sp(species_by_pos, m[2])} {arrow}{m[4]} "
                f"{_STAT.get(m[3], m[3])}")
        elif v == "-sidestart" and len(m) > 3:
            add(f"    hazard/screen up ({m[2][:2]}): {m[3].split(': ')[-1]}")
        elif v == "-sideend" and len(m) > 3:
            add(f"    cleared ({m[2][:2]}): {m[3].split(': ')[-1]}")
        elif v == "-weather" and len(m) > 2 and "upkeep" not in "".join(m[3:]):
            add(f"    weather: {m[2]}")
        elif v == "cant" and len(m) > 3:
            add(f"    {_sp(species_by_pos, m[2])} couldn't move ({m[3]})")
        elif v in ("-fieldstart", "-fieldend") and len(m) > 2:
            add(f"    field {v[1:]}: {m[2].split(': ')[-1]}")
        elif v == "win" and len(m) > 2:
            result = m[2]
    return turns, result, role_us


def build_said(path: str):
    """turn -> list of (persona, text). Preview lines -> turn 0."""
    said: dict[int, list] = {}
    rx = re.compile(r"^\[\s*(T?\d+|--)\]\s+(\S+)\s+(.*)$")
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = rx.match(raw.rstrip("\n"))
        if not m:
            continue
        tag, who, text = m.groups()
        turn = 0 if tag == "--" else int(tag.lstrip("T"))
        said.setdefault(turn, []).append((who, text))
    return said


def write_report(turns, said, result, role_us, username, out):
    with open(out, "w") as f:
        f.write("# GAME (ground truth from protocol)  vs  SAID (commentary)\n")
        f.write(f"# us = {username} = {role_us or '?'}. Commentary [Tn] RECAPS "
                "the exchange going into turn n\n# (so compare SAID [Tn] "
                "against GAME of turn n-1..n).\n\n")
        for t in sorted(set(turns) | set(said)):
            hdr = "PREVIEW / LEADS" if t == 0 else f"TURN {t}"
            f.write(f"\n═══════════ {hdr} ═══════════\n")
            for line in turns.get(t, []):
                f.write(f"  GAME  {line}\n")
            for who, text in said.get(t, []):
                f.write(f"  SAID  {who:<8} {text}\n")
        if result:
            side = "us" if role_us and result == username else "them"
            f.write(f"\n>>> GAME RESULT: {result} won ({side})\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("player_log", help="gen9_player log (--log-level 20)")
    ap.add_argument("transcript", help="capture_feed transcript")
    ap.add_argument("out", help="comparison output path")
    ap.add_argument("--username", default="CBDemo",
                    help="our bot's Showdown username (marks which side is us)")
    args = ap.parse_args()
    turns, result, role_us = build_game(args.player_log, args.username)
    said = build_said(args.transcript)
    write_report(turns, said, result, role_us, args.username, args.out)
    print(f"wrote {args.out}: {len(turns)} game turns, "
          f"{sum(len(v) for v in said.values())} commentary lines"
          + (f", winner {result}" if result else ""))


if __name__ == "__main__":
    main()
