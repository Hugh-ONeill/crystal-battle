#!/usr/bin/env python3
"""Opponent scouting book: per-username profiles mined from OUR OWN game logs.

Ladder opponents repeat hard — one 19-game session faced LLM-gem3f ten times —
so everything we saw last game is free information next game. This turns the
raw logs into per-opponent intelligence:

  roster      the 6-species teams they bring (team preview shows all six, so
              one game teaches their WHOLE team), with counts across games
  sets        per-species revealed moves / items / abilities / tera types
  tendencies  lead choice, tera timing + tera targets, switch rate, game length
  record      our W/L against them

Why it matters beyond reading: a known opponent's known team is a far better
prior than generic usage stats. Phase 1 (this file) produces the book; feeding
it into the translator's set-inference as a top-tier prior is the follow-on.

Usage:
  .venv/bin/python showdown/scouting_book.py --name PAC-Crystal \\
      showdown/bench/overnight_*_ladder.log
  .venv/bin/python showdown/scouting_book.py --name PAC-Crystal --report-only <logs>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ROOM = re.compile(r">(battle-[a-z0-9]+-\d+)")
_PLAYER = re.compile(r"\|player\|(p[12])\|([^|]+)")
_POKE = re.compile(r"\|poke\|(p[12])\|([^,|]+)")
_SWITCH = re.compile(r"\|(?:switch|drag)\|(p[12])a: ([^|]+)\|([^,|]+)")
_MOVE = re.compile(r"\|move\|(p[12])a: ([^|]+)\|([^|]+)")
_ITEM = re.compile(r"\|-(?:item|enditem)\|(p[12])a: ([^|]+)\|([^|]+)")
_ABILITY = re.compile(r"\|-ability\|(p[12])a: ([^|]+)\|([^|]+)")
_TERA = re.compile(r"\|-terastallize\|(p[12])a: ([^|]+)\|(\w+)")
_TURN = re.compile(r"\|turn\|(\d+)")
_WIN = re.compile(r"\|win\|(.+)$")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _species(s: str) -> str:
    """Team preview marks an undisclosed forme with a '-*' suffix
    (Zamazenta-*), which would otherwise count as a 7th roster species."""
    return s.strip().removesuffix("-*").strip()


def parse_battles(paths: list[Path], our_name: str) -> list[dict]:
    """One dict per battle: opponent, their roster/sets, tendencies, result."""
    battles: list[dict] = []
    cur = None
    # nickname -> species, per side (position tokens carry NICKNAMES, which is
    # how prose bugs leak "Speak Softly" instead of the species)
    nick: dict[tuple[str, str], str] = {}

    for path in paths:
        for raw in path.read_text(errors="replace").splitlines():
            line = _ANSI.sub("", raw)

            m = _ROOM.search(line)
            if m and (cur is None or cur["room"] != m.group(1)):
                cur = {"room": m.group(1), "players": {}, "roster": defaultdict(set),
                       "moves": defaultdict(set), "items": {}, "abilities": {},
                       "teras": {}, "tera_turns": {}, "leads": {},
                       "switches": 0, "turn": 0, "winner": None}
                battles.append(cur)
                nick = {}
            if cur is None:
                continue

            m = _PLAYER.search(line)
            if m:
                cur["players"][m.group(1)] = m.group(2).strip()
            m = _TURN.search(line)
            if m:
                cur["turn"] = max(cur["turn"], int(m.group(1)))
            m = _POKE.search(line)
            if m:
                cur["roster"][m.group(1)].add(_species(m.group(2)))
            m = _SWITCH.search(line)
            if m:
                side, nickname = m.group(1), m.group(2).strip()
                species = _species(m.group(3))
                nick[(side, nickname)] = species
                cur["roster"][side].add(species)
                cur["leads"].setdefault(side, species)   # first out, per side
                cur["switches"] += 1
            m = _MOVE.search(line)
            if m:
                side, nickname, move = m.group(1), m.group(2).strip(), m.group(3).strip()
                sp = nick.get((side, nickname), nickname)
                cur["moves"][(side, sp)].add(_norm(move))
            m = _ITEM.search(line)
            if m and "[from] move:" not in line:
                side, nickname, item = m.group(1), m.group(2).strip(), m.group(3).strip()
                sp = nick.get((side, nickname), nickname)
                cur["items"].setdefault((side, sp), _norm(item))
            m = _ABILITY.search(line)
            if m:
                side, nickname, ab = m.group(1), m.group(2).strip(), m.group(3).strip()
                sp = nick.get((side, nickname), nickname)
                cur["abilities"].setdefault((side, sp), _norm(ab))
            m = _TERA.search(line)
            if m:
                side = m.group(1)
                sp = nick.get((side, m.group(2).strip()), m.group(2).strip())
                cur["teras"].setdefault(side, (sp, m.group(3).lower()))
                cur["tera_turns"].setdefault(side, cur["turn"])
            m = _WIN.search(line)
            if m:
                cur["winner"] = m.group(1).strip()

    # orient: keep only battles where we can identify our side + a finished result
    out = []
    for b in battles:
        ours = next((r for r, n in b["players"].items() if n == our_name), None)
        if ours is None or not b["winner"]:
            continue
        opp_side = "p2" if ours == "p1" else "p1"
        b["opp_name"] = b["players"].get(opp_side, "?")
        b["opp_side"] = opp_side
        b["we_won"] = b["winner"] == our_name
        out.append(b)
    return out


def build_book(battles: list[dict]) -> dict:
    book: dict = {}
    for b in battles:
        side = b["opp_side"]
        prof = book.setdefault(b["opp_name"], {
            "games": 0, "our_wins": 0, "rosters": Counter(), "leads": Counter(),
            "sets": defaultdict(lambda: {"moves": Counter(), "items": Counter(),
                                         "abilities": Counter(), "tera": Counter()}),
            "tera_turns": [], "game_lengths": [], "switch_rates": [],
        })
        prof["games"] += 1
        prof["our_wins"] += 1 if b["we_won"] else 0
        roster = tuple(sorted(b["roster"].get(side, [])))
        if len(roster) >= 3:
            prof["rosters"][roster] += 1
        if b["leads"].get(side):
            prof["leads"][b["leads"][side]] += 1
        for (s, sp), mvs in b["moves"].items():
            if s != side:
                continue
            for mv in mvs:
                prof["sets"][sp]["moves"][mv] += 1
        for (s, sp), it in b["items"].items():
            if s == side:
                prof["sets"][sp]["items"][it] += 1
        for (s, sp), ab in b["abilities"].items():
            if s == side:
                prof["sets"][sp]["abilities"][ab] += 1
        if b["teras"].get(side):
            sp, ttype = b["teras"][side]
            prof["sets"][sp]["tera"][ttype] += 1
            if b["tera_turns"].get(side):
                prof["tera_turns"].append(b["tera_turns"][side])
        prof.setdefault("_tera_games", 0)
        prof["_tera_games"] += 1 if b["teras"].get(side) else 0
        if b["turn"]:
            prof["game_lengths"].append(b["turn"])
            prof["switch_rates"].append(b["switches"] / max(1, b["turn"]))
    return book


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def report(book: dict, top: int = 4) -> None:
    for name, p in sorted(book.items(), key=lambda kv: -kv[1]["games"]):
        w, g = p["our_wins"], p["games"]
        print(f"\n=== {name} — we are {w}W-{g - w}L over {g} games "
              f"({w / g:.0%}) ===")
        if p["game_lengths"]:
            print(f"  avg game {_mean(p['game_lengths']):.0f} turns | "
                  f"switch rate {_mean(p['switch_rates']):.2f}/turn"
                  + (f" | tera'd in {p.get('_tera_games', 0)}/{g} games "
                     f"(avg turn {_mean(p['tera_turns']):.0f})"
                     if p["tera_turns"]
                     else f" | NEVER TERA'D in {g} games"))
        if p["leads"]:
            leads = ", ".join(f"{k} x{v}" for k, v in p["leads"].most_common(3))
            print(f"  leads: {leads}")
        for roster, c in p["rosters"].most_common(2):
            print(f"  team x{c}: {', '.join(roster)}")
        shown = 0
        for sp, s in sorted(p["sets"].items(),
                            key=lambda kv: -sum(kv[1]['moves'].values())):
            if not s["moves"] or shown >= top:
                continue
            shown += 1
            mv = ", ".join(m for m, _ in s["moves"].most_common(4))
            extra = []
            if s["items"]:
                extra.append(s["items"].most_common(1)[0][0])
            if s["abilities"]:
                extra.append(s["abilities"].most_common(1)[0][0])
            if s["tera"]:
                extra.append("tera-" + s["tera"].most_common(1)[0][0])
            tail = f"  [{' / '.join(extra)}]" if extra else ""
            print(f"    {sp}: {mv}{tail}")


def _jsonable(book: dict) -> dict:
    out = {}
    for name, p in book.items():
        out[name] = {
            "games": p["games"], "our_wins": p["our_wins"],
            "rosters": [[list(r), c] for r, c in p["rosters"].most_common()],
            "leads": dict(p["leads"]),
            "tera_turns": p["tera_turns"],
            "avg_game_turns": _mean(p["game_lengths"]),
            "avg_switch_rate": _mean(p["switch_rates"]),
            "sets": {sp: {"moves": dict(s["moves"]), "items": dict(s["items"]),
                          "abilities": dict(s["abilities"]),
                          "tera": dict(s["tera"])}
                     for sp, s in p["sets"].items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--name", required=True, help="our username in these logs")
    ap.add_argument("--out", default="showdown/scouting_book.json")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    battles = parse_battles([Path(p) for p in args.logs], args.name)
    book = build_book(battles)
    print(f"parsed {len(battles)} finished games as {args.name}; "
          f"{len(book)} opponents profiled")
    report(book)
    if not args.report_only:
        Path(args.out).write_text(json.dumps(_jsonable(book), indent=1))
        print(f"\nbook written to {args.out}")


if __name__ == "__main__":
    main()
