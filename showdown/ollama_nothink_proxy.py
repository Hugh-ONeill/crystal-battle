#!/usr/bin/env python3
"""Transparent Ollama /v1 proxy that disables model "thinking".

AIRI's chat provider hits Ollama's OpenAI-compatible endpoint
(http://localhost:11434/v1/), and gemma4 is a thinking-capable model that
Ollama runs in thinking mode by default. On the /v1 endpoint there is no
per-request `think` flag (it's ignored), but Ollama 0.32+ maps OpenAI's
`reasoning_effort` — and `reasoning_effort: "none"` turns thinking OFF.

This proxy forwards every request to Ollama UNCHANGED except that it
injects `reasoning_effort: "none"` into chat-completion request bodies.
It does not touch the response at all — streaming SSE and tool-calls pass
through byte-for-byte, so the pokemon_expert tool keeps working.

Point AIRI's consciousness provider base URL at this proxy instead of
Ollama directly:  http://localhost:11435/v1/

Run:  python showdown/ollama_nothink_proxy.py [--port 11435] [--upstream http://localhost:11434]
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://localhost:11434"
_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length",
        "proxy-connection", "te", "trailer", "upgrade"}

import os
_DEBUG = os.environ.get("PROXY_DEBUG") == "1"
_DBG_PATH = "/tmp/claude-1000/-home-wiz/70a27cf4-071e-4963-9faa-1bdda47db203/scratchpad/proxy_debug.log"


def _dbg(msg):
    if _DEBUG:
        try:
            with open(_DBG_PATH, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, method: str):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""

        # the ONLY modification: force thinking off on chat completions
        if method == "POST" and "chat/completions" in self.path and body:
            try:
                obj = json.loads(body)
                if isinstance(obj, dict):
                    obj["reasoning_effort"] = "none"
                    body = json.dumps(obj).encode()
            except Exception:
                pass  # unparseable body: forward verbatim

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in _HOP and k.lower() != "host"}
        req = urllib.request.Request(UPSTREAM + self.path, data=body or None,
                                     method=method, headers=headers)
        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            self.send_error(502, f"upstream error: {e}")
            return

        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in _HOP:
                self.send_header(k, v)
        # Proper HTTP/1.1 chunked framing. The earlier Connection:close +
        # raw-bytes approach worked for curl but AIRI's SSE client (built
        # for Ollama's chunked stream) stopped reading early -> truncated
        # replies. Chunked is the framing it expects.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        total = 0
        try:
            while True:
                chunk = resp.read(2048)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                total += len(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as e:
            _dbg(f"client disconnected after {total}b: {e!r}")
        if _DEBUG and "chat/completions" in self.path:
            try:
                req = json.loads(body)
                keys = sorted(req.keys())
                _dbg(f"chat req keys={keys} stream={req.get('stream')} "
                     f"tools={len(req.get('tools') or [])} -> {total}b resp")
                with open(_DBG_PATH + ".req.json", "w") as rf:
                    rf.write(body.decode("utf-8", "replace"))
            except Exception:
                pass

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")

    def do_DELETE(self):
        self._forward("DELETE")

    def log_message(self, *args):
        pass  # quiet


def main():
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--upstream", default=UPSTREAM)
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ollama no-think proxy: 127.0.0.1:{args.port} -> {UPSTREAM} "
          f"(injects reasoning_effort=none)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
