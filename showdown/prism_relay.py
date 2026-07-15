# Prism in-room commentary relay: log into the local Showdown server as a
# spectator ("Prism") and speak the AIRI character's commentary into the
# battle room chat as it happens — the character appears as a user
# commentating the match live.
#
# Listens on AIRI's server WebSocket for output:gen-ai:chat:complete events
# (each carries BOTH the triggering input text and the reply); only replies
# whose trigger is a battle beat (input starting with "[") are forwarded,
# so unrelated chat with the character never leaks into the room.
#
# Usage (room id from the roomlist query, see scratchpad/room_id.py):
#   .venv/bin/python showdown/prism_relay.py --room battle-gen9ou-1991 \
#       [--username Prism] [--server ws://localhost:8000/showdown/websocket]
#
# Exits ~30s after the battle ends (|win|/|tie| in the room stream), or on
# Ctrl-C. Login mirrors poke-env: challstr -> play.pokemonshowdown.com
# action.php getassertion (no password; falls back to a suffixed name if
# the requested one is registered upstream).

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

from showdown.airi_bridge import DEFAULT_URL as AIRI_URL, _load_token, _unwrap

ACTION_URL = "https://play.pokemonshowdown.com/action.php"
CHAT_LIMIT = 290  # regular-user message cap is 300; keep margin


def _get_assertion(username: str, challstr: str) -> str | None:
    """Guest assertion for an unregistered name; None if name unusable."""
    data = urllib.parse.urlencode({
        "act": "getassertion",
        "userid": "".join(c for c in username.lower() if c.isalnum()),
        "challstr": challstr,
    }).encode()
    req = urllib.request.Request(ACTION_URL, data=data,
                                 headers={"User-Agent": "prism-relay"})
    with urllib.request.urlopen(req, timeout=15) as r:
        assertion = r.read().decode().strip()
    if not assertion or assertion.startswith(";"):
        return None  # registered name (needs password) or refused
    return assertion


_CODE_FENCE = re.compile(r"```.*?```", re.S)
# a leaked feed echo: the model repeats the beat it was given. Everything
# from here on is regurgitation/reasoning, never broadcast copy.
_BEAT_ECHO = re.compile(r"[_*`\s]*\[\s*(?:BATTLE|MATCH|RESULT)\b", re.I)
_NOTE_PAREN = re.compile(r"[_*\s]*\((?:note|aside|correction|edit)\b[^)]*\)[_*]*",
                         re.I)
_META_SENTENCE = re.compile(
    r"(?:^|(?<=[.!?]))\s*[^.!?]*\b(?:let me re-?verify|re-?verify|"
    r"the previous (?:t\d+|turn|line)|i should (?:re-?)?check|as an ai|"
    r"i cannot|i can't help|battle log|ongoing match|as the commentator|"
    r"this (?:is|provides) the (?:battle )?log|my (?:role|task) is)\b"
    r"[^.!?]*[.!?]", re.I)
_SENTENCE = re.compile(r"[^.!?]*[.!?]+")
_MAX_SENTENCES = 3
# thinking-mode tells: once any of these appears the rest is the model's
# scaffolding (self-talk, re-drafts, stage directions), never broadcast copy
_THINKING_TELL = re.compile(
    r"\(\s*(?:done|final|end|executing|ready|proceeding|confidence|"
    r"self-correction|outputting|the end|no more|start outputting|bye)"
    r"|\bActually, I\b|\bFinal (?:Response|Final|answer|decision|String)\b"
    r"|\binternal (?:thought|monologue|dialogue)\b|\bI will just output\b"
    r"|\bI (?:must not|should not) (?:use|say)\b|\bLet me re-?"
    r"|\bLet's go\b|\[END\]|\[Final|\bEnd of thought\b|\bStop thinking\b",
    re.I)


def _sanitize(text: str) -> str:
    """Reduce a raw model reply to clean, spoken broadcast copy.

    Two failure modes to defend against: (1) the character echoes the
    '[BATTLE T8] ...' beat it was fed and rambles about it; (2) with
    thinking mode on, Gemma dumps its whole chain-of-thought ('Final
    Final... Executing... End trace'). A legitimate Prism reply is a single
    line of 1-2 sentences, so: keep only the first paragraph, cut at the
    first feed echo or thinking-tell, drop notes/meta/markdown, and hard-cap
    the sentence count. Nothing multi-paragraph or scaffold-shaped survives."""
    text = _CODE_FENCE.sub("", text)
    text = text.split("\n", 1)[0]           # a real reply is one line
    m = _BEAT_ECHO.search(text)
    if m:
        text = text[:m.start()]
    m = _THINKING_TELL.search(text)
    if m:
        text = text[:m.start()]
    text = _NOTE_PAREN.sub("", text)
    text = _META_SENTENCE.sub("", text)
    text = re.sub(r"[_*`#>]{1,}", "", text)   # markdown emphasis/heading/quote
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _SENTENCE.findall(text)
    if len(sentences) > _MAX_SENTENCES:
        text = "".join(sentences[:_MAX_SENTENCES]).strip()
    return text


def _chunks(text: str, limit: int = CHAT_LIMIT):
    words, cur = text.split(), ""
    for w in words:
        if len(cur) + len(w) + 1 > limit:
            yield cur
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        yield cur


class PrismRelay:
    def __init__(self, room: str, username: str, server: str):
        self.room = room
        self.username = username
        self.server = server
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.battle_over = asyncio.Event()

    async def airi_listener(self):
        """AIRI side: queue every commentary reply triggered by a beat."""
        while not self.battle_over.is_set():
            try:
                async with websockets.connect(AIRI_URL) as ws:
                    await ws.send(json.dumps({
                        "type": "module:authenticate",
                        "data": {"token": _load_token()},
                    }))
                    print("relay: AIRI listener connected", flush=True)
                    async for raw in ws:
                        msg = _unwrap(raw)
                        if msg.get("type") != "output:gen-ai:chat:complete":
                            continue
                        data = msg.get("data", {})
                        beat = (data.get("text") or "").strip()
                        message = data.get("message") or {}
                        cat = message.get("categorization") or {}
                        # prefer the categorized 'speech' field (AIRI's own
                        # extraction of the spoken part, minus reasoning);
                        # fall back to raw content. Sanitize either way.
                        reply = (cat.get("speech")
                                 or message.get("content") or "").strip()
                        if beat.startswith("[") and reply:
                            clean = _sanitize(reply)
                            if clean:
                                self.queue.put_nowait(clean)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"relay: AIRI listener retry ({e!r})", flush=True)
                await asyncio.sleep(3)

    async def showdown_speaker(self):
        """Showdown side: login, join the room, speak queued commentary."""
        async with websockets.connect(self.server) as ws:
            name = await self._login(ws)
            await ws.send(f"|/join {self.room}")
            print(f"relay: {name} joined {self.room}", flush=True)
            reader = asyncio.create_task(self._room_reader(ws))
            try:
                while True:
                    get = asyncio.create_task(self.queue.get())
                    done, _ = await asyncio.wait(
                        {get, reader}, return_when=asyncio.FIRST_COMPLETED)
                    if get in done:
                        text = get.result()
                        for chunk in _chunks(text):
                            await ws.send(f"{self.room}|{chunk}")
                            await asyncio.sleep(0.4)  # PS chat throttle
                        print(f"relay: spoke {len(text)} chars", flush=True)
                    else:
                        get.cancel()
                        break  # room reader ended: battle over + grace
            finally:
                reader.cancel()

    async def _login(self, ws) -> str:
        challstr = None
        while challstr is None:
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            for line in raw.split("\n"):
                if line.startswith("|challstr|"):
                    challstr = line[len("|challstr|"):]
        candidates = [self.username, f"{self.username}Desk",
                      f"{self.username}OnAir"]
        for name in candidates:
            assertion = await asyncio.get_event_loop().run_in_executor(
                None, _get_assertion, name, challstr)
            if assertion:
                await ws.send(f"|/trn {name},0,{assertion}")
                return name
        raise RuntimeError(f"no usable username among {candidates} "
                           "(all registered upstream?)")

    async def _room_reader(self, ws):
        """Watch the room stream; when the battle ends, allow a grace
        window for the final wrap-up lines then finish."""
        async for raw in ws:
            if "|win|" in raw or "|tie|" in raw:
                print("relay: battle ended; 30s grace for the wrap-up",
                      flush=True)
                try:
                    await asyncio.sleep(30)
                finally:
                    self.battle_over.set()
                return

    async def run(self):
        listener = asyncio.create_task(self.airi_listener())
        try:
            await self.showdown_speaker()
        finally:
            listener.cancel()


async def main():
    ap = argparse.ArgumentParser(description="AIRI -> Showdown chat relay")
    ap.add_argument("--room", required=True)
    ap.add_argument("--username", default="Prism")
    ap.add_argument("--server",
                    default="ws://localhost:8000/showdown/websocket")
    args = ap.parse_args()
    relay = PrismRelay(args.room, args.username, args.server)
    await relay.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
