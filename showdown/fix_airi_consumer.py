#!/usr/bin/env python3
"""Fix AIRI's chat-ingestion consumer registration via the Chrome DevTools port.

Upstream race (AIRI 0.11.0): the renderer's server channel reads its auth
token from localStorage (`settings/connection/websocket-auth-token`), which
the tamagotchi settings-apply flow populates from server-channel-config.json
AFTER the channel may already have connected. An unauthenticated peer's
`module:consumer:register` is silently rejected, so every external
`input:text` is dropped with "no consumer registered for event delivery"
in the AIRI log while auth and generation still work.

The fix needs TWO steps, in order:
  1. Copy the settings store's token onto the channel store. That triggers
     the store's [url, token] watch, which disposes and reconnects the
     channel WITH the token. But dispose() resets hasEverConnected, so the
     fresh connection does not count as a "reconnect" and the context
     bridge's onReconnected re-register hook never fires.
  2. Bounce ONLY the context bridge (dispose + initialize) so it re-sends
     its `module:consumer:register` frames on the now-authenticated
     connection. Do NOT dispose the channel manually here: any channel
     bounce after the bridge registers wipes the registration again.

Requires AIRI launched with --remote-debugging-port=9222 (in
~/.config/airi-flags.conf). Exits 0 on success, non-zero on failure.

Usage:  python showdown/fix_airi_consumer.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import websockets

CDP = "http://127.0.0.1:9222/json/list"

PUSH_TOKEN_JS = r"""
(async () => {
  const app = document.querySelector('#app');
  if (!app || !app.__vue_app__) return JSON.stringify({err: 'no vue app'});
  const pinia = app.__vue_app__.config.globalProperties.$pinia;
  const settings = pinia._s.get('tamagotchi-server-channel-settings');
  const chan = pinia._s.get('mods:channels:proj-airi:server');
  if (!settings || !chan) return JSON.stringify({err: 'missing store'});
  try { await settings.refreshServerChannelConfig(); } catch (e) {}
  const token = settings.authToken || '';
  if (!token) return JSON.stringify({err: 'settings token empty'});
  const had = String(chan.websocketAuthToken || '');
  if (had !== token) chan.websocketAuthToken = token;
  return JSON.stringify({token_len: token.length, changed: had !== token});
})()
"""

BOUNCE_BRIDGE_JS = r"""
(async () => {
  const app = document.querySelector('#app');
  const pinia = app.__vue_app__.config.globalProperties.$pinia;
  const chan = pinia._s.get('mods:channels:proj-airi:server');
  const bridge = pinia._s.get('mods:api:context-bridge');
  if (!chan || !bridge) return JSON.stringify({err: 'missing store'});
  try { await chan.ensureConnected(); } catch (e) {}
  try { await bridge.dispose(); } catch (e) {}
  try { await bridge.initialize(); }
  catch (e) { return JSON.stringify({err: 'bridge init: ' + e.message}); }
  await new Promise(r => setTimeout(r, 500));
  return JSON.stringify({connected: !!chan.connected,
                         token_len: String(chan.websocketAuthToken || '').length});
})()
"""


async def _eval(ws, req_id: int, expr: str) -> dict:
    await ws.send(json.dumps({
        "id": req_id, "method": "Runtime.evaluate",
        "params": {"expression": expr, "returnByValue": True,
                   "awaitPromise": True}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == req_id:
            res = msg.get("result", {})
            if "exceptionDetails" in res:
                raise RuntimeError(res["exceptionDetails"].get("text",
                                                               "eval failed"))
            return json.loads(res.get("result", {}).get("value", "{}"))


async def _fix() -> dict:
    with urllib.request.urlopen(CDP, timeout=5) as r:
        targets = json.load(r)
    tgt = next((t for t in targets
                if t["type"] == "page" and "index.html" in t.get("url", "")),
               None)
    if tgt is None:
        raise RuntimeError("AIRI main page not found on the debug port")
    async with websockets.connect(tgt["webSocketDebuggerUrl"],
                                  max_size=50_000_000) as ws:
        pushed = await _eval(ws, 1, PUSH_TOKEN_JS)
        if pushed.get("err"):
            raise RuntimeError(pushed["err"])
        if pushed.get("changed"):
            # let the token watch finish its dispose + reconnect cycle
            await asyncio.sleep(3)
        bounced = await _eval(ws, 2, BOUNCE_BRIDGE_JS)
        if bounced.get("err"):
            raise RuntimeError(bounced["err"])
        return {**pushed, **bounced}


def main():
    try:
        print("airi consumer fix:", asyncio.run(_fix()))
    except Exception as e:
        print(f"consumer fix failed ({e!r}); is AIRI running with "
              "--remote-debugging-port=9222?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
