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
import random
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _merge_mcts_results(results) -> list:
    """Combine side_one results from searches over different sampled
    opponent worlds: sum visits and scores per move_choice, rank by visits.
    A move that only looks good in one world loses to one that holds up
    across all of them."""
    merged: dict[str, SimpleNamespace] = {}
    for result in results:
        for r in result.side_one:
            m = merged.get(r.move_choice)
            if m is None:
                merged[r.move_choice] = SimpleNamespace(
                    move_choice=r.move_choice,
                    visits=r.visits, total_score=r.total_score)
            else:
                m.visits += r.visits
                m.total_score += r.total_score
    return sorted(merged.values(), key=lambda m: -m.visits)


def _preview_order(lead_idx: int, n: int) -> str:
    """'/team 312456'-style order string: chosen lead first, rest in order."""
    rest = [i for i in range(1, n + 1) if i != lead_idx + 1]
    return "/team " + "".join(str(x) for x in [lead_idx + 1] + rest)


def _lead_pool(matrix, epsilon: float = 0.08) -> list[int]:
    """Lead indices whose maximin (worst-case row value) is within epsilon
    of the best. A deterministic maximin lead is optimally predictable — we
    measured 30/30 identical leads per series, a free read for the opponent.
    Sampling among near-ties keeps the choice sound but unreadable."""
    row_mins = [min(row) for row in matrix]
    best = max(row_mins)
    return [i for i, v in enumerate(row_mins) if v >= best - epsilon]


def _select_choice(mappable, rng, sample: bool = True, keep_ratio: float = 0.75):
    """Pick from visit-ranked mappable candidates. Argmax is exploitable:
    foul-play keeps every move >= 75% of its best and samples — same rule
    here. `mappable` is a non-empty list of (result, order) tuples sorted
    by visits descending."""
    if not sample or len(mappable) == 1:
        return mappable[0]
    top = mappable[0][0].visits
    pool = [m for m in mappable if m[0].visits >= keep_ratio * top]
    weights = [m[0].visits for m in pool]
    return rng.choices(pool, weights=weights)[0]


class Gen9PokeEnginePlayer(Player):
    """poke-env Player: translate -> poke-engine MCTS -> order.

    The MCTS result is a visit-ranked list of engine choices; we walk it
    best-first and play the first choice that maps onto something the
    server actually offered (available_moves / available_switches /
    can_tera). That one loop handles tera legality, choice lock, Encore,
    trapping, and disabled moves without format-specific branches.
    """

    def __init__(self, search_ms: int = 1000, set_source: str = "gen9ou",
                 team_paste: str | None = None, preview_search_ms: int = 80,
                 set_samples: int = 2, data_tiers: bool = True,
                 stochastic: bool = True, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._translator = Gen9Translator(set_source=set_source,
                                          use_data_tiers=data_tiers)
        self._stochastic = stochastic
        self._choice_rng = random.Random()
        self._search_ms = search_ms
        self._team_paste = team_paste
        self._preview_search_ms = preview_search_ms
        # >1: search that many sampled opponent-set worlds per turn and merge
        # (chaos sources only; monotype canonical sets have no sampler yet)
        self._set_samples = set_samples if set_source not in (None, "monotype") else 1
        self._verbose = verbose
        self._last_tag: str | None = None

    async def teampreview(self, battle):
        """6x6 MCTS maximin over (our lead, their predicted lead) pairings —
        a fixed lead hands the opponent a free, certain counter-pick every
        game. Falls back to paste order on any failure."""
        if self._team_paste is None:
            return "/team 123456"
        try:
            from monotype.lead_picker import pick_leads
            opp_species = [m.species for m in battle.opponent_team.values()]
            opp_paste = self._translator.predicted_preview_paste(opp_species)
            loop = asyncio.get_event_loop()
            lead_idx, _, matrix = await loop.run_in_executor(
                None, lambda: pick_leads(self._team_paste, opp_paste,
                                         search_ms=self._preview_search_ms))
            pool = _lead_pool(matrix)
            if self._stochastic and len(pool) > 1:
                lead_idx = self._choice_rng.choice(pool)
            order = _preview_order(lead_idx, 6)
            if self._verbose:
                print(f"  preview: leading slot {lead_idx + 1} "
                      f"(pool of {len(pool)}) -> {order}")
            return order
        except Exception as e:
            if self._verbose:
                print(f"  preview pick failed ({e!r}); using paste order")
            return "/team 123456"

    async def choose_move(self, battle):
        if battle.battle_tag != self._last_tag:
            self._last_tag = battle.battle_tag
            self._translator.new_battle()

        # forced switch (post-KO / pivot): search it like any other decision —
        # the translator flags side_one.force_switch and (for KOs) leaves the
        # fainted active at slot 0, so MCTS returns replacement choices
        if battle.force_switch:
            if not battle.available_switches:
                return self.choose_default_move()
            if len(battle.available_switches) == 1:
                return self.create_order(battle.available_switches[0])
            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, self._search_samples, battle)
                order = self._map_choice(_merge_mcts_results(results), battle)
                if order is not None:
                    return order
            except Exception as e:
                if self._verbose:
                    print(f"  T{battle.turn} force-switch search failed "
                          f"({e!r}); using heuristic")
            best = max(battle.available_switches,
                       key=lambda p: _score_switch_in(p, battle))
            return self.create_order(best)

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._search_samples, battle)
        except Exception as e:
            if self._verbose:
                print(f"  T{battle.turn} translate/search failed ({e!r}); "
                      f"choosing randomly")
            return self.choose_random_move(battle)

        ranked = _merge_mcts_results(results)
        order = self._map_choice(ranked, battle)
        if order is not None:
            return order
        if self._verbose:
            print(f"  T{battle.turn} no MCTS choice mapped to a legal order; "
                  f"choosing randomly")
        return self.choose_random_move(battle)

    def _search_samples(self, battle) -> list:
        """One MCTS per sampled opponent world (K = set_samples). With K=1,
        the deterministic top-set translation is used, as before. The LAST
        world is speed-pessimistic (fastest spreads, scarf when plausible):
        speed-floor inference only triggers after a scarfer already outsped
        something, so one world hedges against the sweep pre-emptively."""
        results = []
        for i in range(self._set_samples):
            rng = random.Random() if self._set_samples > 1 else None
            pessimistic = self._set_samples > 1 and i == self._set_samples - 1
            # world 0: curated PS joint sets; later worlds: chaos sampling.
            # PS sets in every world collapsed diversity (series 10) — some
            # species have a single curated candidate, so all worlds shared
            # the same confident wrong set
            prefer_ps = i == 0
            state = self._translator.translate(
                battle, rng=rng, speed_pessimistic=pessimistic,
                prefer_ps=prefer_ps)
            results.append(pe.monte_carlo_tree_search(state, self._search_ms))
        return results

    def _map_choice(self, ranked, battle):
        """Collect every legal engine choice, then pick: probabilistic among
        near-ties (>=75% of the top's visits) when stochastic, else argmax."""
        moves_by_id = {_normalize(m.id): m for m in battle.available_moves}
        switches_by_id = {_normalize(p.species): p
                          for p in battle.available_switches}
        mappable = []
        for r in ranked:
            choice = r.move_choice
            if choice.startswith("switch "):
                target = switches_by_id.get(_normalize(choice[7:]))
                if target is not None:
                    mappable.append((r, self.create_order(target),
                                     f"switch {target.species}"))
                continue
            tera = choice.endswith("-tera")
            move_id = _normalize(choice[:-5] if tera else choice)
            move = moves_by_id.get(move_id)
            if move is None:
                continue
            if tera and not battle.can_tera:
                continue  # engine explored tera we don't have; try next
            mappable.append((r, self.create_order(move, terastallize=tera),
                             move_id + (" (tera)" if tera else "")))
        if not mappable:
            return None
        chosen_result, (order, desc) = _select_choice(
            [(m[0], (m[1], m[2])) for m in mappable],
            self._choice_rng, sample=self._stochastic)
        self._log_choice(battle, chosen_result, desc)
        return order

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
    parser.add_argument("--set-samples", type=int, default=2,
                        help="sampled opponent-set worlds searched per turn "
                             "(1 = deterministic top sets)")
    parser.add_argument("--data-tiers", choices=["on", "off"], default="on",
                        help="PS-curated + replay-observed set tiers; 'off' "
                             "reproduces the pure chaos config (ab9 baseline)")
    parser.add_argument("--stochastic", choices=["on", "off"], default="on",
                        help="sample moves among near-ties (75%% rule) and "
                             "leads among maximin near-ties; 'off' = argmax")
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
        team_paste=team,
        set_samples=args.set_samples,
        data_tiers=args.data_tiers == "on",
        stochastic=args.stochastic == "on",
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
