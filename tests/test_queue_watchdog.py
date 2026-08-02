"""Queue watchdog: frees an unmatched ladder slot without ever touching a
matched one. The decouple lets PER_GAME_TIMEOUT stay generous (2700s — 900s
was killing live richwoman grinds as disconnect forfeits) while a dead
queue recycles the slot in minutes."""
import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_player import _queue_watchdog


def test_unmatched_slot_bails_after_deadline():
    player = SimpleNamespace(_battles={})

    async def run():
        await asyncio.wait_for(
            _queue_watchdog(player, deadline_s=0.15, interval=0.02),
            timeout=2.0)
    asyncio.run(run())          # returns (no TimeoutError) = bail fired


def test_matched_slot_parks_forever():
    player = SimpleNamespace(_battles={"battle-gen9ou-1": object()})

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                _queue_watchdog(player, deadline_s=0.15, interval=0.02),
                timeout=0.6)    # well past the deadline: it must NOT return
    asyncio.run(run())


def test_late_match_still_parks():
    player = SimpleNamespace(_battles={})

    async def run():
        task = asyncio.ensure_future(
            _queue_watchdog(player, deadline_s=0.3, interval=0.02))
        await asyncio.sleep(0.1)
        player._battles["battle-gen9ou-2"] = object()   # match mid-wait
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.6)
    asyncio.run(run())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
