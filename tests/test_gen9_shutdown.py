"""Clean-exit watchdog.

poke-env's listen() swallows an abnormal websocket close (a 1011 keepalive
ping timeout is caught by its bare `except Exception` and only logged), then
stops receiving — so accept_challenges()/ladder() hang forever on battles that
can't finish and the process wedges with its MCTS trees resident (~5-6 GB
leaked per hung run this session, swap exhaustion, OOM of the next launch).
_ws_closed_watchdog must return once the socket is CLOSED so main() can bail,
and must NOT fire while it's OPEN or before the socket exists."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_player import _ws_closed_watchdog


def _player(state_name):
    ws = (SimpleNamespace(state=SimpleNamespace(name=state_name))
          if state_name else None)
    return SimpleNamespace(ps_client=SimpleNamespace(websocket=ws))


def test_returns_on_closed_socket():
    asyncio.run(asyncio.wait_for(_ws_closed_watchdog(_player("CLOSED"), 0.01),
                                 2))


def test_waits_while_open():
    async def go():
        try:
            await asyncio.wait_for(
                _ws_closed_watchdog(_player("OPEN"), 0.01), 0.3)
            raise AssertionError("watchdog fired while the socket was OPEN")
        except asyncio.TimeoutError:
            pass
    asyncio.run(go())


def test_handles_preconnect_no_socket():
    async def go():
        try:
            await asyncio.wait_for(
                _ws_closed_watchdog(_player(None), 0.01), 0.3)
            raise AssertionError("watchdog fired before the socket existed")
        except asyncio.TimeoutError:
            pass
    asyncio.run(go())


def test_state_enum_has_closed():
    # the .name == "CLOSED" check is pinned to the real websockets enum
    from websockets.protocol import State
    assert State.CLOSED.name == "CLOSED"


def test_active_ability_single_known_only():
    """Only a DEFINITE ability is handed to the caster for fact injection:
    our own mon resolves to one; an unrevealed opponent (several dex
    candidates) stays None so PRISM is never fed a guess as fact."""
    from showdown.gen9_player import Gen9PokeEnginePlayer as P
    fake = SimpleNamespace(
        _ability_lookup=lambda name, side: (
            {"goodasgold"} if side == "us" else {"levitate", "clearbody"}))
    assert P._active_ability(fake, "Gholdengo", "us") == "goodasgold"
    assert P._active_ability(fake, "Unrevealed", "them") is None  # ambiguous
    assert P._active_ability(fake, None, "us") is None

    def boom(name, side):
        raise RuntimeError("dex miss")
    assert P._active_ability(
        SimpleNamespace(_ability_lookup=boom), "X", "us") is None


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
