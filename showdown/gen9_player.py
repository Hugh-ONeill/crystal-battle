# Gen 9 live player: poke-env Player driving poke-engine MCTS through the
# gen9 translator (showdown/gen9_translator.py).
#
# Primary target is gen9ou on the PokeAgent server (pokeagentshowdown.com,
# the living benchmark whose 2025 gen9 OU bracket was won by stock foul-play,
# this project's upstream). Also plays gen9monotype: the server never offers
# tera there (Terastal Clause), so "-tera" search choices simply fail to map
# and the next-best candidate is used.
#
# Usage (local sparring vs foul-play):
#   .venv/bin/python showdown/gen9_player.py --local --username CBGen9 \
#       --mode challenge --user-to-challenge FPSpar1 \
#       --format gen9ou --team teams/gen9ou_sample.txt --n-games 1
#
# PokeAgent (bot usernames should start with "PAC"):
#   .venv/bin/python showdown/gen9_player.py --server pokeagent \
#       --username PAC-Crystal9 --password ... --mode ladder --format gen9ou

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration

from showdown.name_mapping import _normalize
from showdown.gen9_translator import Gen9Translator
from showdown.poke_engine_player import _score_switch_in

LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

POKEAGENT_SERVER = ServerConfiguration(
    "wss://pokeagentshowdown.com/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

SERVERS = {
    "local": LOCAL_SERVER,
    "pokeagent": POKEAGENT_SERVER,
    "showdown": ShowdownServerConfiguration,
}


class Gen9PokeEnginePlayer(Player):
    """poke-env Player: translate -> poke-engine MCTS -> order.

    The MCTS result is a visit-ranked list of engine choices; we walk it
    best-first and play the first choice that maps onto something the
    server actually offered (available_moves / available_switches /
    can_tera). That one loop handles tera legality, choice lock, Encore,
    trapping, and disabled moves without format-specific branches.
    """

    def __init__(self, search_ms: int = 1000, set_source: str = "gen9ou",
                 verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._translator = Gen9Translator(set_source=set_source)
        self._search_ms = search_ms
        self._verbose = verbose
        self._last_tag: str | None = None

    def teampreview(self, battle):
        # paste order. TODO: MCTS-eval preview picker (the trained lead net
        # is monotype-only; don't use it here)
        return "/team 123456"

    async def choose_move(self, battle):
        if battle.battle_tag != self._last_tag:
            self._last_tag = battle.battle_tag
            self._translator.new_battle()

        # forced switch (post-KO / pivot): matchup-scored heuristic pick
        if battle.force_switch:
            if battle.available_switches:
                best = max(battle.available_switches,
                           key=lambda p: _score_switch_in(p, battle))
                return self.create_order(best)
            return self.choose_default_move()

        try:
            state = self._translator.translate(battle)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, pe.monte_carlo_tree_search, state, self._search_ms)
        except Exception as e:
            if self._verbose:
                print(f"  T{battle.turn} translate/search failed ({e!r}); "
                      f"choosing randomly")
            return self.choose_random_move(battle)

        ranked = sorted(result.side_one, key=lambda r: -r.visits)
        order = self._map_choice(ranked, battle)
        if order is not None:
            return order
        if self._verbose:
            print(f"  T{battle.turn} no MCTS choice mapped to a legal order; "
                  f"choosing randomly")
        return self.choose_random_move(battle)

    def _map_choice(self, ranked, battle):
        """First visit-ranked engine choice that the server allows."""
        moves_by_id = {_normalize(m.id): m for m in battle.available_moves}
        switches_by_id = {_normalize(p.species): p
                          for p in battle.available_switches}
        for r in ranked:
            choice = r.move_choice
            if choice.startswith("switch "):
                target = switches_by_id.get(_normalize(choice[7:]))
                if target is not None:
                    self._log_choice(battle, r, f"switch {target.species}")
                    return self.create_order(target)
                continue
            tera = choice.endswith("-tera")
            move_id = _normalize(choice[:-5] if tera else choice)
            move = moves_by_id.get(move_id)
            if move is None:
                continue
            if tera and not battle.can_tera:
                continue  # engine explored tera we don't have; try next
            self._log_choice(battle, r, move_id + (" (tera)" if tera else ""))
            return self.create_order(move, terastallize=tera)
        return None

    def _log_choice(self, battle, r, desc: str):
        if self._verbose:
            print(f"  T{battle.turn}: {desc} "
                  f"(visits={r.visits}, "
                  f"avg_score={r.total_score / max(1, r.visits):.3f})")


async def main():
    parser = argparse.ArgumentParser(description="gen9 poke-engine live player")
    parser.add_argument("--local", action="store_true",
                        help="shorthand for --server local")
    parser.add_argument("--server", choices=list(SERVERS), default="local")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--format", dest="fmt", default="gen9ou")
    parser.add_argument("--team", default=None,
                        help="path to a Showdown paste file (required for "
                             "team formats)")
    parser.add_argument("--search-ms", type=int, default=1000)
    parser.add_argument("--set-source", default=None,
                        help="opponent set inference source; defaults to "
                             "'monotype' for gen9monotype else the format name")
    parser.add_argument("--mode", choices=["accept", "challenge", "ladder"],
                        default="accept")
    parser.add_argument("--user-to-challenge", default=None)
    parser.add_argument("--n-games", type=int, default=1)
    parser.add_argument("--log-level", type=int, default=30,
                        help="poke-env logger level (10=DEBUG shows protocol)")
    args = parser.parse_args()

    server = LOCAL_SERVER if args.local else SERVERS[args.server]
    team = Path(args.team).read_text() if args.team else None
    set_source = args.set_source or (
        "monotype" if args.fmt == "gen9monotype" else args.fmt)

    player = Gen9PokeEnginePlayer(
        search_ms=args.search_ms,
        set_source=set_source,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=server,
        battle_format=args.fmt,
        team=team,
        max_concurrent_battles=1,
        # NOTE: accept_open_team_sheet=True makes poke-env defer the team
        # preview reply until a showteam/rejection message that formats
        # without the OTS rule never send -> guaranteed timer loss
        log_level=args.log_level,
    )

    if args.mode == "challenge":
        if not args.user_to_challenge:
            parser.error("--mode challenge requires --user-to-challenge")
        await player.send_challenges(args.user_to_challenge,
                                     n_challenges=args.n_games)
    elif args.mode == "ladder":
        await player.ladder(args.n_games)
    else:
        await player.accept_challenges(None, args.n_games)

    print(f"finished: {player.n_won_battles}W / "
          f"{player.n_lost_battles}L / {player.n_tied_battles}T")


if __name__ == "__main__":
    asyncio.run(main())
