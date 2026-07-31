#!/usr/bin/env python3
"""Render roles.json as a readable review document.

roles.json is the source of truth and where edits land; this is a generated
view for reading and annotating. Regenerate rather than editing the markdown,
or the two drift and the prose silently becomes the wrong version.

  .venv/bin/python showdown/roles_report.py            # to stdout
  .venv/bin/python showdown/roles_report.py -o ROLES.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
GRADE = {"measured": "measured here",
         "rag-grounded": "Smogon-cited",
         "user-corrected": "USER-CORRECTED",
         "unmeasured": "UNVERIFIED"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def usage_map() -> dict[str, float]:
    try:
        raw = json.loads((HERE / "gen9ou_chaos.json").read_text())
        data = raw.get("data", raw)
        return {norm(k): v.get("usage", 0.0) for k, v in data.items()}
    except Exception:
        return {}


def render(doc: dict) -> str:
    roles, use = doc["roles"], usage_map()
    order = sorted(roles.items(), key=lambda kv: -use.get(kv[0], 0.0))
    out: list[str] = []
    w = out.append

    w("# Role annotations — review copy\n")
    w(f"Generated from `showdown/roles.json` ({len(roles)} entries). "
      "**Edit the JSON, not this file.**\n")
    w("Nothing consumes these yet. **The plain paragraph under each entry is the "
      "`fact`** — the only field a consumer would ever be shown. It is written to "
      "stand alone: present tense, mechanically true, no named opponents, no "
      "references to other entries. Everything about where a claim came from lives "
      "in the collapsed provenance block, which no consumer sees.\n")
    w("| grade | meaning |")
    w("|---|---|")
    w("| measured here | backed by a measurement from this campaign |")
    w("| Smogon-cited | grounded in retrieved Smogon analysis |")
    w("| USER-CORRECTED | you corrected my reading — highest confidence |")
    w("| UNVERIFIED | my inference only — scrutinise these |\n")

    w("## At a glance\n")
    w("| usage | mon | grade | preserve | deployment | tags |")
    w("|---:|---|---|---|---|---|")
    for k, v in order:
        u = use.get(k, 0.0)
        cond = " ⚡" if v.get("conditional") else ""
        multi = " ✳" if v.get("sets") else ""
        seq = " ▸" if v.get("sequence") else ""
        w(f"| {100*u:.1f}% | **{k}**{cond}{multi}{seq} | {GRADE.get(v['review'], v['review'])} "
          f"| {v.get('preserve','—')} | {v.get('deployment','—')} | {', '.join(v['tags'])} |")
    w("\n⚡ = value is conditional  ✳ = splits into distinct sets  "
      "▸ = has a written play sequence\n")

    w("## Entries\n")
    for k, v in order:
        u = use.get(k, 0.0)
        w(f"### {k} — {100*u:.1f}% usage · *{GRADE.get(v['review'], v['review'])}*\n")
        bits = [f"**tags** {', '.join(v['tags'])}"]
        for f in ("ability", "preserve", "deployment", "lead_intent",
                  "entry_condition", "value_curve", "resource", "requires"):
            if v.get(f):
                bits.append(f"**{f}** {v[f]}")
        w(" · ".join(bits) + "\n")
        if v.get("fact"):
            w(v["fact"] + "\n")

        for st in v.get("sets", []):
            bits = [f"**tags** {', '.join(st['tags'])}"]
            for f in ("preserve", "deployment"):
                if st.get(f):
                    bits.append(f"**{f}** {st[f]}")
            w(f"**Set — {st['name']}:** " + " · ".join(bits) + "\n")
            if st.get("fact"):
                w(f"> {st['fact']}\n")

        if v.get("sequence"):
            w("**The play:**\n")
            for n, step in enumerate(v["sequence"], 1):
                w(f"{n}. {step}")
            if v.get("sequence_note"):
                w(f"\n> {v['sequence_note']}")
            w("")

        for cnd in v.get("conditional", []):
            w(f"**Conditional — `{cnd['when']}`:** "
              + " · ".join(f"{f} → {cnd[f]}" for f in ("preserve", "tags")
                           if cnd.get(f)))
            for f in ("note", "usage"):
                if cnd.get(f):
                    w(f"\n> {cnd[f]}")
            w("")

        w(f"<details><summary>provenance (review only — not shown to any consumer)"
          f"</summary>\n\n{v['provenance']}\n\n</details>\n")

    unver = [k for k, v in roles.items() if v["review"] == "unmeasured"]
    if unver:
        w(f"## Scrutinise first\n\nUnverified (my inference only): "
          f"{', '.join(unver)}\n")
    w("## Known schema gap\n")
    w("`preserve` for a hazard remover (Great Tusk, Corviknight) depends on "
      "whether **our own** team has another remover — a team property this "
      "per-species file cannot express. `conditional` only covers opponent "
      "facts. Needs a team-level pass before anything consumes it.\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("--roles", default=str(HERE / "roles.json"))
    args = ap.parse_args()
    text = render(json.loads(Path(args.roles).read_text()))
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(text.splitlines())} lines)", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
