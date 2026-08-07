#!/usr/bin/env python3
"""Gate the OU lead net against the baseline it would REPLACE.

Val top-1 on held-out replays says the net learned something. It does not say
the net is better than what `_lead_ev_blend` uses today, which is the scouting
book's per-opponent lead counter. This scores both predictors on the SAME
events — our real ladder games against booked opponents — where the book is at
its strongest (richwoman alone has 454 games of evidence).

Three predictors, same 6-way choice, restricted to the roster actually
previewed:
  net       argmax of the lead net, from both previews alone
  book      argmax of that opponent's observed lead counter (production)
  prior     argmax of the field-wide lead frequency (opponent-agnostic)

The net's whole claim is GENERALISATION: the book cannot predict an unseen
October entrant at all, while the net needs only their preview. So beating the
book on booked opponents is the hard case, not the easy one.

Usage: .venv/bin/python showdown/eval_ou_lead_net.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from monotype.featurizer_lead_preview import featurize_preview  # noqa: E402
from showdown.extract_ou_lead_data import roster_to_paste  # noqa: E402
from showdown.ps_sets import get_index  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def ladder_previews(name: str = "PAC-Crystal"):
    """(opponent, our_roster, their_roster, their_lead) per ladder game."""
    out = []
    for f in sorted(glob.glob(str(HERE / "bench" / "overnight_*_ladder.log"))):
        room = me = opp_name = None
        roster = defaultdict(list)
        lead = {}
        for line in open(f, errors="replace"):
            m = re.search(r">(battle-gen9oulongtimer-\d+)", line)
            if m:
                if m.group(1) != room:
                    if me and opp_name:
                        o = "p2" if me == "p1" else "p1"
                        if len(roster[o]) == 6 and len(roster[me]) == 6 \
                                and lead.get(o):
                            out.append((opp_name, roster[me], roster[o],
                                        lead[o]))
                    room = m.group(1)
                    me = opp_name = None
                    roster = defaultdict(list)
                    lead = {}
                continue
            if not line.startswith("|") or room is None:
                continue
            p = line.rstrip().split("|")
            k = p[1] if len(p) > 1 else ""
            if k == "player" and len(p) > 3:
                if p[3].strip() == name:
                    me = p[2]
                else:
                    opp_name = p[3].strip()
            elif k == "poke" and len(p) > 3:
                roster[p[2]].append(p[3].split(",")[0].strip())
            elif k == "switch" and len(p) > 3:
                s = p[2][:2]
                if s not in lead:
                    lead[s] = p[3].split(",")[0].strip()
    return out


def main():
    import torch
    from monotype.lead_net import LeadPickerNet

    book = json.loads((HERE / "scouting_book.json").read_text())
    index = get_index("gen9ou")
    ckpt = torch.load(HERE / "ou_lead_net.pt", map_location="cpu",
                      weights_only=False)
    net = LeadPickerNet()
    net.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    net.eval()

    # field-wide prior: how often each species is led when present
    field = Counter()
    had = Counter()
    games = ladder_previews()
    for _opp, _ours, theirs, ld in games:
        for sp in theirs:
            had[_norm(sp)] += 1
        field[_norm(ld)] += 1

    hits = Counter()
    n = 0
    per_opp = defaultdict(lambda: Counter())
    for opp, ours, theirs, ld in games:
        prof = book.get(opp)
        if not prof:
            continue
        truth = _norm(ld)
        if truth not in {_norm(s) for s in theirs}:
            continue
        p_ours = roster_to_paste(ours, index)
        p_theirs = roster_to_paste(theirs, index)
        if not p_ours or not p_theirs:
            continue
        n += 1
        # net: featurise from THEIR view (their team first)
        try:
            f_them, f_us = featurize_preview(p_theirs, p_ours)
            with torch.no_grad():
                logits = net(torch.tensor(f_them).unsqueeze(0),
                             torch.tensor(f_us).unsqueeze(0))
            pick = int(torch.argmax(logits, dim=-1).item())
            net_ok = _norm(theirs[pick]) == truth
        except Exception:
            continue
        # book: their observed lead counter, restricted to today's roster
        counts = {_norm(k): v for k, v in (prof.get("leads") or {}).items()}
        bk = max(theirs, key=lambda s: counts.get(_norm(s), 0))
        book_ok = _norm(bk) == truth
        # field prior
        fp = max(theirs, key=lambda s: field.get(_norm(s), 0) /
                 max(had.get(_norm(s), 1), 1))
        prior_ok = _norm(fp) == truth
        hits["net"] += net_ok
        hits["book"] += book_ok
        hits["prior"] += prior_ok
        per_opp[opp]["n"] += 1
        per_opp[opp]["net"] += net_ok
        per_opp[opp]["book"] += book_ok

    if not n:
        sys.exit("no scoreable ladder previews")
    print(f"scoreable ladder games vs booked opponents: {n}\n")
    print(f"{'predictor':10s} {'top-1':>8}")
    for k in ("net", "book", "prior"):
        print(f"{k:10s} {hits[k]/n:8.1%}")
    print(f"\n{'opponent':16s} {'n':>5} {'net':>8} {'book':>8}")
    for opp, c in sorted(per_opp.items(), key=lambda kv: -kv[1]["n"]):
        if c["n"] < 15:
            continue
        print(f"{opp:16s} {c['n']:5d} {c['net']/c['n']:8.1%} "
              f"{c['book']/c['n']:8.1%}")


if __name__ == "__main__":
    main()
