#!/usr/bin/env python3
"""Clear AIRI's chat-session history via the Chrome DevTools port.

The commentator streams one input:text per battle beat into AIRI's chat,
and AIRI sends the WHOLE session history on every request. Left unchecked
that history grows without bound (hundreds of messages across matches) and
overflows the model's context window, at which point replies truncate
after a few words. Clear the session before each match to keep the context
small.

Requires AIRI launched with --remote-debugging-port=9222 (in
~/.config/airi-flags.conf). No-op-safe: prints a status and exits 0 on
success, non-zero if AIRI/CDP isn't reachable.

Usage:  python showdown/clear_airi_history.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import websockets

CDP = "http://127.0.0.1:9222/json/list"
CLEAR_JS = r"""
(async () => {
  const app = document.querySelector('#app');
  if (!app || !app.__vue_app__) return 'no vue app';
  const pinia = app.__vue_app__.config.globalProperties.$pinia;
  const cs = pinia._s.get('chat-session');
  if (!cs) return 'no chat-session store';
  const sid = cs.activeSessionId;
  const before = cs.getSessionMessages ? (cs.getSessionMessages(sid) || []).length : '?';
  try {
    if (typeof cs.setSessionMessages === 'function' && sid) cs.setSessionMessages(sid, []);
    if (typeof cs.persistSessionMessages === 'function' && sid) await cs.persistSessionMessages(sid);
  } catch (e) { return 'error: ' + e.message; }
  const after = cs.getSessionMessages ? (cs.getSessionMessages(sid) || []).length : '?';
  return 'cleared ' + before + ' -> ' + after;
})()
"""


async def _clear() -> str:
    with urllib.request.urlopen(CDP, timeout=5) as r:
        targets = json.load(r)
    tgt = next((t for t in targets
                if t["type"] == "page" and "index.html" in t.get("url", "")),
               None)
    if tgt is None:
        raise RuntimeError("AIRI main page not found on the debug port")
    async with websockets.connect(tgt["webSocketDebuggerUrl"],
                                  max_size=50_000_000) as ws:
        await ws.send(json.dumps({
            "id": 1, "method": "Runtime.evaluate",
            "params": {"expression": CLEAR_JS, "returnByValue": True,
                       "awaitPromise": True}}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1:
                res = msg.get("result", {})
                if "exceptionDetails" in res:
                    raise RuntimeError(res["exceptionDetails"].get("text",
                                                                   "eval failed"))
                return res.get("result", {}).get("value", "no result")


def main():
    try:
        print("airi history:", asyncio.run(_clear()))
    except Exception as e:
        print(f"clear failed ({e!r}); is AIRI running with "
              "--remote-debugging-port=9222?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
