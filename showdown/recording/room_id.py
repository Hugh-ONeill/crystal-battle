"""Print an active battle room id on the local Showdown server.

WITH --player NAME, print only a room that NAME is playing in. Use it.

The unfiltered form takes rooms[0], i.e. whichever battle the server happens to
list first. That is only safe on an idle server. With a bench series running
(par_series lanes churn hundreds of rooms) it silently returns someone else's
game, and the failure is invisible downstream: the broadcast composite happily
spectates a bench lane while the commentary narrates OUR match, so picture and
captions describe different battles and every claim looks like a
misattribution. Cost a contaminated presentation-clock measurement on
2026-07-27 before it was spotted.
"""
import argparse
import asyncio
import json

import websockets


async def main(player=None):
    want = player.lower().replace(" ", "") if player else None
    for _ in range(25):
        try:
            async with websockets.connect(
                    "ws://localhost:8000/showdown/websocket") as ws:
                await ws.send('|/cmd roomlist')
                end = asyncio.get_event_loop().time() + 4
                while asyncio.get_event_loop().time() < end:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                    if "roomlist" not in raw:
                        continue
                    for line in raw.split("\n"):
                        if "queryresponse|roomlist" not in line:
                            continue
                        data = json.loads(line.split("|", 3)[3])
                        rooms = data.get("rooms", {}) or {}
                        if want:
                            for rid, info in rooms.items():
                                names = [
                                    str((info or {}).get(k, ""))
                                    .lower().replace(" ", "")
                                    for k in ("p1", "p2")
                                ]
                                if want in names:
                                    print(rid)
                                    return
                        elif rooms:
                            print(list(rooms)[0])
                            return
        except Exception:
            pass
        await asyncio.sleep(2)
    print("NO-BATTLE-FOUND")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None,
                    help="only match a battle this user is playing in")
    args = ap.parse_args()
    asyncio.run(main(args.player))
