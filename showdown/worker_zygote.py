#!/usr/bin/env python3
"""Prefork zygote for bench workers: load once, fork many.

Each bench worker independently loads ~1.1GB of identical read-only data
(measured 2026-07-25: replay-sets corroboration index 768M, chaos stats
~150M, PS-set index ~54M) — 16 workers = 17GB and the commentary demo
cannot coexist. This process warms those module-level singleton caches
ONCE, calls gc.freeze() (Instagram's permanent-generation trick — without
it CPython refcount writes dirty the copy-on-write pages), then forks
workers on request. Children inherit the warm caches through COW; their
normal get_index() calls find them populated, so gen9_player needs no
changes at all.

Post-fork discipline (the sharp edges):
  - env from the request is applied BEFORE anything touches the engine, so
    CB_* OnceLocks read the per-arm values (env-flag A/B arms share one
    zygote; PYTHONPATH build-shadow arms CANNOT — the module is already
    loaded — par_series refuses that combination);
  - random.seed() re-rolls from OS entropy — forked children otherwise
    share the parent's PRNG state and would sample identical worlds;
  - event loops, websockets, and ONNX sessions are created by the child's
    own main(); nothing stateful exists pre-fork.

Protocol: par_series writes the PATH of a request file into the FIFO.
Request file lines: `resp=<file>`, `log=<file>`, zero or more `env K=V`,
then `arg <token>` per argv token. The zygote forks, the child redirects
stdio to the log and runs gen9_player.main(); the zygote writes the child
pid to the resp file. par_series kills workers by that pid exactly as
before; a SIGCHLD-driven waitpid loop reaps them.
"""

from __future__ import annotations

import gc
import os
import signal
import sys
import time

CB = "/home/wiz/Developer/grimoire/crystal-battle"


def _warm_caches(set_source: str) -> None:
    import poke_engine  # noqa: F401  (module import only; no engine calls)
    from showdown.ps_sets import get_index as ps_get
    from showdown.replay_sets import get_index as rp_get
    from showdown.chaos_stats import ChaosStats
    from showdown.gen9_translator import Gen9Translator

    t0 = time.time()
    ps_get(set_source)
    rp_get(set_source)
    Gen9Translator._chaos_cache[set_source] = ChaosStats(format=set_source)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print(f"zygote warm: {rss}M resident in {time.time() - t0:.1f}s",
          flush=True)


def _quiesce_poke_env_loop() -> None:
    """poke-env starts a daemon event-loop thread (concurrency.POKE_LOOP driven by
    _t) at IMPORT; _warm_caches imports poke-env via ps_sets, so the zygote would be
    MULTI-THREADED at os.fork() -> children inherit a driverless/locked POKE_LOOP and
    deadlock nondeterministically (the flaky half-launch bug). Stop the thread here
    so the fork is single-threaded; each child rebuilds a fresh loop in _restart."""
    try:
        import atexit
        import poke_env.concurrency as pc
        pc.POKE_LOOP.call_soon_threadsafe(pc.POKE_LOOP.stop)
        pc._t.join(timeout=5)
        if pc._t.is_alive():
            print("zygote: poke-env loop thread did not stop in 5s", flush=True)
        # poke-env registers atexit __clear_loop, which assumes a RUNNING loop; on
        # our stopped loop its shutdown_asyncgens future never resolves -> HANG at
        # a clean exit. Best-effort unregister it, and CLOSE the loop so that even
        # if unregister misses, the handler errors-and-continues instead of hanging.
        try:
            atexit.unregister(getattr(pc, "__clear_loop"))
        except Exception:
            pass
        pc.POKE_LOOP.close()
    except Exception as e:
        print(f"zygote: could not quiesce poke-env loop: {e!r}", flush=True)


def _restart_poke_env_loop() -> None:
    """Rebuild poke-env's global loop + driver thread in the forked CHILD: the
    inherited POKE_LOOP is dead (its thread didn't survive the fork) and its
    selector fds are shared with the parent, so a fresh loop is mandatory.
    Player/PSClient capture POKE_LOOP as an __init__ DEFAULT ARG (bound at import),
    so reassigning the module global is NOT enough -- repoint those defaults at the
    new loop too, or Players silently drive the dead one."""
    import asyncio
    import threading
    import poke_env.concurrency as pc
    old = pc.POKE_LOOP
    new = asyncio.new_event_loop()

    def _drive(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_drive, args=(new,), daemon=True).start()
    pc.POKE_LOOP = new
    try:
        import poke_env.player.player as _pp
        import poke_env.ps_client.ps_client as _pc
        for fn in (_pp.Player.__init__, _pc.PSClient.__init__):
            # `loop` is a KEYWORD-ONLY arg -> its default is in __kwdefaults__
            # (a dict), not __defaults__; handle both to be version-robust.
            if fn.__defaults__:
                fn.__defaults__ = tuple(
                    new if d is old else d for d in fn.__defaults__)
            if fn.__kwdefaults__:
                for k, v in fn.__kwdefaults__.items():
                    if v is old:
                        fn.__kwdefaults__[k] = new
    except Exception as e:
        print(f"child: could not repoint poke-env loop defaults: {e!r}", flush=True)


def _spawn(req_path: str) -> None:
    resp = log = None
    env: dict[str, str] = {}
    argv: list[str] = []
    for line in open(req_path):
        line = line.rstrip("\n")
        if line.startswith("resp="):
            resp = line[5:]
        elif line.startswith("log="):
            log = line[4:]
        elif line.startswith("env "):
            k, _, v = line[4:].partition("=")
            env[k] = v
        elif line.startswith("arg "):
            argv.append(line[4:])
    if not (resp and log and argv):
        return
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.environ.update(env)
            _restart_poke_env_loop()  # inherited poke-env loop is dead + fd-shared
            import random
            random.seed()  # fork copies the parent PRNG state; re-roll
            sys.argv = ["gen9_player.py"] + argv
            import asyncio
            from showdown.gen9_player import main as player_main
            asyncio.run(player_main())
            os._exit(0)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
    with open(resp, "w") as fh:
        fh.write(str(pid))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fifo", required=True)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--set-source", default="gen9ou")
    args = ap.parse_args()

    sys.path.insert(0, CB)
    os.chdir(CB)
    _warm_caches(args.set_source)
    _quiesce_poke_env_loop()  # single-threaded fork: stop poke-env's daemon loop
    gc.freeze()

    # reap exited workers so they don't accumulate as zombies
    signal.signal(signal.SIGCHLD, lambda *_: None)

    if not os.path.exists(args.fifo):
        os.mkfifo(args.fifo)
    with open(args.ready_file, "w") as fh:
        fh.write(str(os.getpid()))
    print(f"zygote ready on {args.fifo}", flush=True)

    while True:
        try:
            with open(args.fifo) as fh:  # blocks until a writer connects
                for line in fh:
                    req = line.strip()
                    if not req:
                        continue
                    if req == "shutdown":
                        print("zygote shutting down", flush=True)
                        return 0
                    try:
                        _spawn(req)
                    except Exception as e:
                        print(f"spawn failed for {req}: {e!r}", flush=True)
        except InterruptedError:
            pass
        finally:
            # opportunistic reap
            try:
                while os.waitpid(-1, os.WNOHANG)[0] > 0:
                    pass
            except ChildProcessError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
