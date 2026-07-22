"""Decompose per-game wall time from a gen9_player ours.log.

Usage: python showdown/decompose_game_time.py showdown/bench/<run>_L*_ours.log

Every protocol line is timestamped to the ms. Attribute each inter-line gap
to whoever ENDED it: a gap ending in a '>>>' line is time WE spent (search +
translate + featurize); a gap ending in '<<<' is time spent WAITING on the
server/opponent. Battle window = |init|battle .. |win|. Also measures startup
(process start -> battle init) and turns per game.
"""
import re
import sys
from datetime import datetime

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - \S+ - \w+ - (.*)")


def records(path):
    """Yield (timestamp, full_record_text) — protocol messages span lines."""
    t, buf = None, []
    with open(path, errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if m:
                if t is not None:
                    yield t, "\n".join(buf)
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
                buf = [m.group(2)]
            elif t is not None:
                buf.append(line.rstrip("\n"))
    if t is not None:
        yield t, "\n".join(buf)


def parse(path):
    games, cur, prev_t, start_t = [], None, None, None
    for t, body in records(path):
            if "Starting listening" in body:
                start_t = t
            if "|init|battle" in body:
                cur = dict(t0=t, us=0.0, wait=0.0, turns=0, our_events=0,
                           startup=(t - start_t).total_seconds() if start_t else None)
                prev_t = t
            elif cur is not None:
                d = (t - prev_t).total_seconds()
                if body.startswith("\x1b[93m") or ">>>" in body[:20]:
                    cur["us"] += d
                    cur["our_events"] += 1
                else:
                    cur["wait"] += d
                cur["turns"] += body.count("|turn|")
                prev_t = t
                if "|win|" in body:
                    cur["wall"] = (t - cur["t0"]).total_seconds()
                    games.append(cur)
                    cur = None
    return games


def report(label, games):
    if not games:
        print(f"{label}: no complete games parsed")
        return
    n = len(games)
    def avg(k): return sum(g[k] for g in games) / n
    tot_turns = sum(g["turns"] for g in games)
    tot_us = sum(g["us"] for g in games)
    tot_wait = sum(g["wait"] for g in games)
    tot_wall = sum(g["wall"] for g in games)
    startups = [g["startup"] for g in games if g["startup"] is not None]
    print(f"{label}: {n} games, {tot_turns} turns")
    print(f"  battle wall/game : {avg('wall'):7.1f}s   turns/game: {avg('turns'):5.1f}")
    print(f"  our time/turn    : {tot_us/tot_turns*1000:7.0f}ms  ({tot_us/tot_wall*100:4.1f}% of battle wall)")
    print(f"  wait time/turn   : {tot_wait/tot_turns*1000:7.0f}ms  ({tot_wait/tot_wall*100:4.1f}% of battle wall)")
    print(f"  sec/turn overall : {tot_wall/tot_turns:7.2f}s")
    if startups:
        print(f"  startup->battle  : {sum(startups)/len(startups):7.1f}s avg over {len(startups)} fresh starts")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        label = path.split("/")[-1]
        report(label, parse(path))
