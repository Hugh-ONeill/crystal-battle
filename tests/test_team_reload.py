"""--team-reload wiring: a persistent accept-mode worker must pick up the
team the harness swapped in — both for /utm (ReloadingTeambuilder) and for
the lead picker (_team_paste refresh at team preview).

The timing test matters most: poke-env's accept loop pre-yields the team at
iteration start, i.e. BEFORE the harness swaps the file for the next game,
so the refresh must re-run at challenge receipt (which is always after the
swap). Caught live before this hook existed: a lane's second game /utm'd
its first game's team."""

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from showdown.gen9_player import Gen9PokeEnginePlayer, ReloadingTeambuilder

PIKA = """Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
"""

EEVEE = """Eevee @ Eviolite
Ability: Adaptability
EVs: 252 HP / 252 Atk
Adamant Nature
- Quick Attack
"""


class TestReloadingTeambuilder(unittest.TestCase):
    def test_yield_reflects_file_swap(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "lane.team"
            path.write_text(PIKA)
            tb = ReloadingTeambuilder(str(path))
            self.assertIn("pikachu", tb.yield_team().lower())
            path.write_text(EEVEE)
            packed = tb.yield_team().lower()
            self.assertIn("eevee", packed)
            self.assertNotIn("pikachu", packed)


class TestTeamPasteRefresh(unittest.TestCase):
    def _fake(self, reload_path, paste="stale"):
        return SimpleNamespace(_team_reload_path=reload_path,
                               _team_paste=paste)

    def test_refresh_reads_current_file(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "lane.team"
            path.write_text(PIKA)
            fake = self._fake(str(path))
            Gen9PokeEnginePlayer._refresh_team_paste(fake)
            self.assertEqual(fake._team_paste, PIKA)
            path.write_text(EEVEE)
            Gen9PokeEnginePlayer._refresh_team_paste(fake)
            self.assertEqual(fake._team_paste, EEVEE)

    def test_no_reload_path_is_a_noop(self):
        fake = self._fake(None, paste="original")
        Gen9PokeEnginePlayer._refresh_team_paste(fake)
        self.assertEqual(fake._team_paste, "original")


class TestChallengeTimeRefresh(unittest.TestCase):
    """_handle_challenge_request must re-yield the packed team so the
    /utm sent on accept reflects the file as of the CHALLENGE, not as of
    the previous battle's end."""

    def _worker(self, path):
        p = Gen9PokeEnginePlayer.__new__(Gen9PokeEnginePlayer)
        p._team_reload_path = str(path)
        p._team = ReloadingTeambuilder(str(path))
        p._current_packed_team = "stale-previous-game"
        # challenger == our username makes poke-env's own handler a no-op,
        # isolating the refresh
        p.ps_client = SimpleNamespace(username="me")
        return p

    def test_challenge_refreshes_packed_team(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "lane.team"
            path.write_text(PIKA)
            p = self._worker(path)
            asyncio.run(p._handle_challenge_request(["", "pm", "me"]))
            self.assertIn("pikachu", p._current_packed_team.lower())
            path.write_text(EEVEE)
            asyncio.run(p._handle_challenge_request(["", "pm", "me"]))
            self.assertIn("eevee", p._current_packed_team.lower())

    def test_without_reload_path_keeps_poke_env_team(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "lane.team"
            path.write_text(PIKA)
            p = self._worker(path)
            p._team_reload_path = None
            asyncio.run(p._handle_challenge_request(["", "pm", "me"]))
            self.assertEqual(p._current_packed_team, "stale-previous-game")


if __name__ == "__main__":
    unittest.main()
