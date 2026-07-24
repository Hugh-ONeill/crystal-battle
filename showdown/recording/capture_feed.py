#!/usr/bin/env python3
"""Subscribe to the commentary feed (commentary_overlay on :8130) and write a
readable transcript — silent, no per-beat output. Read the whole file at match
end: reviewing the transcript catches hallucinations and mis-attributions that
are invisible watching live per-beat. Companion to compare.py.

Run:  .venv/bin/python -m showdown.recording.capture_feed [out.txt] [--url ...]
"""
from __future__ import annotations

import argparse
import asyncio
import json

import websockets

DEFAULT_URL = "ws://127.0.0.1:8130"


def format_line(d: dict) -> str | None:
    """One feed payload -> one transcript line, or None to skip (no text)."""
    text = (d.get("text") or "").strip()
    if not text:
        return None
    turn = d.get("turn")
    who = d.get("persona") or "-"
    tag = f"T{turn}" if turn is not None else "--"
    moment = f"  [{d['moment']}]" if d.get("moment") else ""
    board = ""
    if d.get("us") and d.get("them"):
        board = (f"  ({d['us']} {d.get('us_hp', '?')}% vs "
                 f"{d['them']} {d.get('them_hp', '?')}%"
                 f" | {d.get('us_alive', '?')}-{d.get('them_alive', '?')})")
    return f"[{tag:>4}] {who:<8} {text}{moment}{board}"


async def run(out: str, url: str):
    with open(out, "w") as f:
        f.write("")  # truncate
    async for ws in websockets.connect(url):
        try:
            # the feed replays its last cached line to every new subscriber, so
            # the first frame after connecting is stale (a prior game's beat at
            # a fresh start, or a dup we already captured on a reconnect). A
            # real beat never arrives this fast — drain the instant replay.
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                pass  # no cached line waiting (fresh feed)
            async for msg in ws:
                try:
                    d = json.loads(msg)
                except Exception:
                    continue
                line = format_line(d)
                if line is None:
                    continue
                with open(out, "a") as f:
                    f.write(line + "\n")
        except websockets.ConnectionClosed:
            continue


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("out", nargs="?", default="demo_transcript.txt",
                    help="transcript output path (truncated on start)")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="overlay feed websocket (commentary_overlay)")
    args = ap.parse_args()
    asyncio.run(run(args.out, args.url))


if __name__ == "__main__":
    main()
