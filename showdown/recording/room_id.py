"""Print the first active battle room id on the local Showdown server."""
import asyncio
import json

import websockets


async def main():
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
                        if "queryresponse|roomlist" in line:
                            data = json.loads(line.split("|", 3)[3])
                            rooms = list(data.get("rooms", {}))
                            if rooms:
                                print(rooms[0])
                                return
        except Exception:
            pass
        await asyncio.sleep(2)
    print("NO-BATTLE-FOUND")


asyncio.run(main())
