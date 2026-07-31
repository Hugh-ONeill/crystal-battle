"""Match closeness from three independent signals, joined per battle:
  1. root eval trajectory (desk_reads jsonl: per-turn engine self-assessment)
  2. final material margin (faints from the protocol)
  3. turn length
"""
import json, re, sys, glob
from pathlib import Path

def results_from_log(path):
    """battle_tag -> (result, our_faints, their_faints, turns, opp)"""
    out = {}
    cur = None; us = None; oppside = None
    faints = {}; turns = {}; opp = {}; res = {}
    for line in open(path, errors="ignore"):
        m = re.search(r">(battle-gen9oulongtimer-\d+)", line)
        if m:
            cur = m.group(1)
        if cur is None:
            continue
        pm = re.match(r"\|player\|(p[12])\|([^|]+)", line.strip())
        if pm:
            if pm.group(2) == "PAC-Crystal":
                us = pm.group(1)
                # record per-room
                opp.setdefault(cur, {})["us"] = pm.group(1)
            else:
                opp.setdefault(cur, {})["opp"] = pm.group(2)
                opp[cur]["oppside"] = pm.group(1)
        tm = re.match(r"\|turn\|(\d+)", line.strip())
        if tm:
            turns[cur] = int(tm.group(1))
        fm = re.match(r"\|faint\|(p[12])a:", line.strip())
        if fm:
            faints.setdefault(cur, {"p1": 0, "p2": 0})[fm.group(1)] += 1
        wm = re.match(r"\|win\|(.+)", line.strip())
        if wm:
            res[cur] = "W" if wm.group(1).strip() == "PAC-Crystal" else "L"
    for tag, r in res.items():
        info = opp.get(tag, {})
        u = info.get("us"); f = faints.get(tag, {"p1": 0, "p2": 0})
        if not u:
            continue
        ours = f.get(u, 0); theirs = f.get("p2" if u == "p1" else "p1", 0)
        out[tag] = (r, ours, theirs, turns.get(tag, 0), info.get("opp", "?"))
    return out

def main(stamp):
    log = f"showdown/bench/overnight_{stamp}_ladder.log"
    desk = f"showdown/desk_reads_{stamp}.jsonl"
    res = results_from_log(log)
    rows = []
    for line in open(desk, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        tag = d["battle_tag"]
        if tag not in res:
            continue
        r, ours, theirs, turns, opp = res[tag]
        reads = [v for _, v in d["reads"]]
        if not reads:
            continue
        rows.append(dict(tag=tag, r=r, opp=d.get("opponent", opp), turns=turns,
                         ours=ours, theirs=theirs,
                         peak=max(reads), final=sum(reads[-3:]) / len(reads[-3:]),
                         mean=sum(reads) / len(reads)))
    if not rows:
        print("  no joined rows"); return
    def agg(sel, label):
        s = [x for x in rows if sel(x)]
        if not s: return
        n = len(s)
        print(f"  {label:22s} n={n:3d}  peak-eval {sum(x['peak'] for x in s)/n:.2f}  "
              f"final-eval {sum(x['final'] for x in s)/n:.2f}  "
              f"mons-left us {6-sum(x['ours'] for x in s)/n:.1f} them {6-sum(x['theirs'] for x in s)/n:.1f}  "
              f"turns {sum(x['turns'] for x in s)/n:.0f}")
    print(f"=== {stamp} ({len(rows)} games with eval trajectories) ===")
    agg(lambda x: x['r']=='W', "WINS")
    agg(lambda x: x['r']=='L', "LOSSES")
    agg(lambda x: x['r']=='L' and x['opp']=='richwoman', "  losses vs richwoman")
    agg(lambda x: x['r']=='L' and x['opp']!='richwoman', "  losses vs bots")
    L = [x for x in rows if x['r']=='L']
    if L:
        blowout = [x for x in L if x['peak'] < 0.60]
        thrown  = [x for x in L if x['peak'] >= 0.80]
        close   = [x for x in L if 6-x['theirs'] <= 2]
        print(f"  loss shape: never-ahead(peak<0.60) {len(blowout)}/{len(L)} ({100*len(blowout)//len(L)}%)  "
              f"thrown(peak>=0.80) {len(thrown)}  "
              f"close-at-end(<=2 mons left them) {len(close)}")

for s in sys.argv[1:]:
    main(s); print()
