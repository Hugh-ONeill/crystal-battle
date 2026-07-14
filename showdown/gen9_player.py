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
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


_TIME_LEFT_RE = re.compile(r"has (\d+) sec")
_TIME_TOTAL_RE = re.compile(r"\|\s*(\d+) sec total")


def _time_left(battle, username: str | None) -> int | None:
    """Seconds left in OUR bank, parsed from the server's |inactive| clock
    messages in the retained protocol history, or None if unknown. Two
    formats: '<user> has N seconds left.' (per-player countdown) and
    'Time left: X sec this turn | Y sec total'. Returns the most recent."""
    replay = getattr(battle, "_replay_data", None)
    if not replay:
        return None
    latest = None
    uname = (username or "").lower()
    for event in replay:
        if len(event) < 3 or event[1] != "inactive":
            continue
        msg = event[2]
        low = msg.lower()
        if uname and low.startswith(uname) and "sec" in low:
            m = _TIME_LEFT_RE.search(msg)
            if m:
                latest = int(m.group(1))
        elif "total" in low:
            m = _TIME_TOTAL_RE.search(msg)
            if m:
                latest = int(m.group(1))
    return latest


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
                 stochastic: bool = True, adaptive: bool = False,
                 escalate_ms: int = 2000, flat_threshold: float = 0.55,
                 clock_floor_s: int = 40, escalate_bank_s: float = 90.0,
                 value_net_path: str | None = None, value_alpha: float = 0.5,
                 value_batch: int = 32, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._translator = Gen9Translator(set_source=set_source,
                                          use_data_tiers=data_tiers)
        self._stochastic = stochastic
        self._choice_rng = random.Random()
        # concurrent search across sampled worlds (real parallelism once the
        # mcts binding's GIL release is built; harmless serialization before)
        self._search_pool = ThreadPoolExecutor(max_workers=max(1, set_samples))
        # optional learned leaf eval: mcts_with_value blends the value net
        # with the static eval by alpha (0=static, 1=pure net). ~3.7x fewer
        # iterations even batched, so this only wins where static-eval
        # blindness (flat stall positions) wastes plain MCTS's iterations.
        self._value_net = None
        if value_net_path:
            self._value_net = pe.ValueNet(value_net_path)
            print(f"loaded value net {value_net_path} "
                  f"(alpha={value_alpha}, batch={value_batch})")
        self._value_alpha = value_alpha
        self._value_batch = value_batch
        # adaptive search: probe at search_ms, escalate to escalate_ms in
        # flat (undecided) positions. In stall games flat is the NORM, so
        # escalating every flat turn blows the clock; a per-game bank of
        # extra seconds (self-tracked — the local server doesn't emit the
        # |inactive| timer messages) caps total spend and concentrates it on
        # the earliest/most-contested positions. The parsed server clock is
        # an additional safety when present.
        self._adaptive = adaptive
        self._probe_ms = search_ms
        self._escalate_ms = escalate_ms
        self._flat_threshold = flat_threshold
        self._clock_floor_s = clock_floor_s
        self._escalate_bank_s = escalate_bank_s
        self._bank_used_s = 0.0
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
            self._bank_used_s = 0.0  # fresh escalation bank per game

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
            search = self._adaptive_search if self._adaptive else self._search_samples
            results = await loop.run_in_executor(None, search, battle)
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

    def _search_samples(self, battle, search_ms: int | None = None) -> list:
        """One MCTS per sampled opponent world (K = set_samples). With K=1,
        the deterministic top-set translation is used, as before. The LAST
        world is speed-pessimistic (fastest spreads, scarf when plausible):
        speed-floor inference only triggers after a scarfer already outsped
        something, so one world hedges against the sweep pre-emptively.

        Translation must be serial (the translator mutates per-call instance
        state: rng/pessimism/prefer_ps/archetype/obs), but the searches are
        independent and the mcts binding releases the GIL (py.detach), so
        they run concurrently across cores when set_samples > 1. Identical
        results either way — only wall time differs."""
        ms = search_ms if search_ms is not None else self._search_ms
        states = []
        for i in range(self._set_samples):
            rng = random.Random() if self._set_samples > 1 else None
            pessimistic = self._set_samples > 1 and i == self._set_samples - 1
            # world 0: curated PS joint sets; later worlds: chaos sampling.
            # PS sets in every world collapsed diversity (series 10) — some
            # species have a single curated candidate, so all worlds shared
            # the same confident wrong set
            prefer_ps = i == 0
            states.append(self._translator.translate(
                battle, rng=rng, speed_pessimistic=pessimistic,
                prefer_ps=prefer_ps))
        search = self._search_one
        if len(states) == 1:
            return [search(states[0], ms)]
        return list(self._search_pool.map(lambda st: search(st, ms), states))

    def _search_one(self, state, ms: int):
        """One world's search: value-net-guided leaf eval when a net is
        loaded, else plain MCTS."""
        if self._value_net is None:
            return pe.monte_carlo_tree_search(state, ms)
        return pe.monte_carlo_tree_search_with_value(
            state, self._value_net, ms,
            alpha=self._value_alpha, batch_size=self._value_batch)

    def _adaptive_search(self, battle) -> list:
        """Staged search that reinvests the timer bank where it matters.

        Fixed 300ms/decision leaves ~90% of the server clock unused in the
        long grindy games (stall/fat), which are exactly the positions where
        a flat static eval makes every move look equal and deeper search is
        most likely to break the tie. So: probe cheap; if the merged visit
        distribution is DECISIVE (one move dominates — a resolved tactic),
        keep it. If it's FLAT and the clock is healthy, re-search deep and
        use that instead. Sharp positions self-select out (they produce
        peaked distributions at the probe budget); quiet/attrition positions
        produce flat ones and get the extra thinking."""
        probe = self._search_samples(battle, self._probe_ms)
        merged = _merge_mcts_results(probe)
        total = sum(m.visits for m in merged) or 1
        top_share = merged[0].visits / total if merged else 1.0
        if top_share >= self._flat_threshold or len(merged) <= 1:
            return probe  # decisive — a resolved tactic, don't spend more

        # flat position: escalate only while the per-game bank holds, and
        # (when the server clock is visible) while it's healthy
        if self._bank_used_s >= self._escalate_bank_s:
            return probe
        try:
            uname = self.username
        except Exception:
            uname = None
        clock = _time_left(battle, uname)
        if clock is not None and clock <= self._clock_floor_s:
            return probe
        budget = self._escalate_ms
        if clock is not None:
            safe = max(self._probe_ms,
                       int((clock - self._clock_floor_s) * 1000 * 0.25
                           / max(1, self._set_samples)))
            budget = min(budget, safe)
        if budget <= self._probe_ms:
            return probe
        if self._verbose:
            print(f"  T{battle.turn} flat (top {top_share:.0%}), escalating "
                  f"{self._probe_ms}->{budget}ms/world "
                  f"(bank {self._bank_used_s:.0f}/{self._escalate_bank_s:.0f}s)")
        t0 = time.monotonic()
        deep = self._search_samples(battle, budget)
        self._bank_used_s += time.monotonic() - t0
        return deep

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
    parser.add_argument("--adaptive", choices=["on", "off"], default="off",
                        help="staged search: probe at --search-ms, escalate to "
                             "--escalate-ms in flat positions when clock allows")
    parser.add_argument("--escalate-ms", type=int, default=2000,
                        help="deep-search budget per world in flat positions")
    parser.add_argument("--escalate-bank-s", type=float, default=90.0,
                        help="per-game budget of extra seconds for escalation")
    parser.add_argument("--value-net", type=str, default=None,
                        help="path to a ValueNet ONNX for leaf eval "
                             "(mcts_with_value); omit for pure static eval")
    parser.add_argument("--value-alpha", type=float, default=0.5,
                        help="value-net blend weight (0=static, 1=pure net)")
    parser.add_argument("--value-batch", type=int, default=32,
                        help="leaf-eval batch size (throughput vs quality)")
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
        adaptive=args.adaptive == "on",
        escalate_ms=args.escalate_ms,
        escalate_bank_s=args.escalate_bank_s,
        value_net_path=args.value_net,
        value_alpha=args.value_alpha,
        value_batch=args.value_batch,
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
