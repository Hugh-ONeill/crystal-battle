#!/usr/bin/env python3
"""Predictable-policy sparring bot for the local rig.

Plays gen9ou by SAMPLING the human opponent-policy net (opp_policy_human_v1.pt)
as its OWN move policy. Two jobs, both blocked on the local rig having only
foul-play as an opponent:

  (a) EXPLOITABILITY — the opponent-policy s2-prior lever came back neutral vs
      foul-play because fp is near-optimal and unmodelable (min-max search wants
      a paranoid opponent model; imitation gives typical-case; a near-optimal
      opponent has nothing to exploit). This bot plays the SAME human
      distribution our opp-net was trained on, so vs it our priors are a near-
      perfect model -> the cleanest possible test that "predicting the opponent
      converts to winrate" when the prediction is actually good.

  (b) CRASH-FREE ESCALATION — stock foul-play throws KeyError 'battle' when our
      turns run long (grind-depth escalation), which blocks the grind-depth
      winrate A/B. This bot never parses our protocol, so it can't choke on a
      slow turn.

Orientation: the net predicts SIDE_TWO's action. To predict the bot's OWN
action we translate the battle normally (bot = side_one), then MIRROR the state
(swap side_one<->side_two) so the bot sits in side_two, featurize the mirror,
and map the 14 slot classes [m0-3, m0-3-tera, party0-5] back onto the legal
poke-env orders. A temperature controls how sharply the bot commits to the
net's argmax (T->0 deterministic, T=1 = the net's own distribution).

NOT for the PokeAgent ladder or any self-sparring win-trade: local server only.
Ad-hoc smoke names use Sparbot*, never CBGen9/FPSpar1 (username collision kicks
a running series worker).
"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

# let `python showdown/spar_bot.py` resolve the `showdown` package (matches
# gen9_player.py) — must precede the package imports below
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import poke_engine as pe
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration

from showdown.gen9_translator import Gen9Translator
from showdown.opp_policy_gate import _load_net
from showdown.opp_policy_train import _flip  # exact training-time state flip
from showdown.featurizer_v3 import parse_state_v3
from showdown.name_mapping import _normalize

LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

DEFAULT_NET = os.path.join(os.path.dirname(__file__), "opp_policy_human_v1.pt")

# 14 slot classes the net emits for side_two, in order.
_N_MOVE, _N_TERA_OFF, _N_SWITCH_OFF = 4, 4, 8


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class SparBot(Player):
    """poke-env Player that samples the human opponent-policy net as its policy."""

    def __init__(self, net_path: str = DEFAULT_NET, temperature: float = 1.0,
                 set_source: str = "gen9ou", seed: int | None = None,
                 verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._net = _load_net(net_path)
        self._temp = max(1e-3, float(temperature))
        self._translator = Gen9Translator(set_source=set_source)
        self._rng = random.Random(seed)
        self._verbose = verbose
        self._last_tag = None
        print(f"SparBot up: net={os.path.basename(net_path)} "
              f"temperature={self._temp} set_source={set_source}", flush=True)

    def _battle_finished_callback(self, battle):
        # Emit foul-play's exact tally line ("INFO     Winner: <username>") so
        # par_series's grep-based tally counts spar-vs-CB games with no harness
        # change: CB wins -> "Winner: CBGen9...", spar wins -> our username.
        if battle.won is True:
            winner = self._me()
        elif battle.won is False:
            winner = getattr(battle, "opponent_username", "?")
        else:
            winner = "tie"
        print(f"INFO     Winner: {winner}", flush=True)

    def _me(self) -> str:
        u = getattr(self, "username", None)
        if u:
            return u
        ac = getattr(self, "account_configuration", None)
        return getattr(ac, "username", "?") if ac else "?"

    def teampreview(self, battle):
        # lead choice is not what this bot exists to test; keep it random so it
        # doesn't leak a fixed lead our own lead-picker could over-fit to.
        return self.random_teampreview(battle)

    def choose_move(self, battle):
        if battle.battle_tag != self._last_tag:
            self._last_tag = battle.battle_tag
            self._translator.new_battle()
        try:
            order = self._net_order(battle)
            if order is not None:
                return order
        except Exception as e:
            if self._verbose:
                print(f"  T{getattr(battle, 'turn', '?')} net decision failed "
                      f"({e!r}); random", flush=True)
        return self.choose_random_move(battle)

    def _net_order(self, battle):
        moves = list(battle.available_moves)
        switches = list(battle.available_switches)
        if not moves and not switches:
            return self.choose_default_move()

        # translate normally (bot = side_one), then FLIP the state string so the
        # bot lands in side_two — the exact orientation the net was trained to
        # predict (opp_policy_train augments every record with _flip -> s1). Slot
        # identities come straight off state.side_one (== the bot post-flip),
        # so no object reconstruction is needed.
        state = self._translator.translate(battle)
        logits = self._forward(_flip(state.to_string()))
        dist = _softmax(logits)

        side = state.side_one  # the bot; == flipped-state side_two
        active = side.pokemon[int(side.active_index)]
        move_ids = [_normalize(str(active.moves[i].id))
                    for i in range(min(_N_MOVE, len(active.moves)))]
        species = [_normalize(str(p.id)) for p in side.pokemon]
        move_slot = {mid: i for i, mid in enumerate(move_ids)}
        species_slot = {sp: i for i, sp in enumerate(species)}

        cands = []  # (order, prob)
        for m in moves:
            i = move_slot.get(_normalize(m.id))
            base = dist[i] if i is not None else 0.0
            cands.append((self.create_order(m, terastallize=False), base))
            if battle.can_tera and i is not None:
                cands.append((self.create_order(m, terastallize=True),
                              dist[_N_TERA_OFF + i]))
        for p in switches:
            j = species_slot.get(_normalize(p.species))
            prob = dist[_N_SWITCH_OFF + j] if j is not None else 0.0
            cands.append((self.create_order(p), prob))

        if not cands:
            return None
        return self._sample(cands)

    def _forward(self, state_str: str) -> np.ndarray:
        import torch
        x = torch.from_numpy(parse_state_v3(state_str)).float().unsqueeze(0)
        with torch.no_grad():
            return self._net(x)[0].numpy()

    def _sample(self, cands):
        # temperature-tilt the net probs, keep a small floor so a legal action
        # the net zeroed is still reachable (mirrors the search's uniform floor)
        eps = 1e-6
        w = np.array([max(eps, float(p)) for _, p in cands], dtype=np.float64)
        w = w ** (1.0 / self._temp)
        s = w.sum()
        w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
        idx = self._rng.choices(range(len(cands)), weights=w.tolist(), k=1)[0]
        return cands[idx][0]


async def main():
    ap = argparse.ArgumentParser(description="human-policy sparring bot")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", default=None)
    ap.add_argument("--net", default=DEFAULT_NET)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--set-source", default="gen9ou")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--team", default=None, help="team paste file")
    ap.add_argument("--mode", choices=["accept", "challenge", "ladder"],
                    default="accept")
    ap.add_argument("--user-to-challenge", default=None)
    ap.add_argument("--n-games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    team = open(args.team).read() if args.team else None
    bot = SparBot(
        net_path=args.net, temperature=args.temperature,
        set_source=args.set_source, seed=args.seed, verbose=args.verbose,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=LOCAL_SERVER,
        battle_format=args.format, team=team, max_concurrent_battles=1,
    )

    if args.mode == "challenge":
        if not args.user_to_challenge:
            ap.error("--mode challenge requires --user-to-challenge")
        await bot.send_challenges(args.user_to_challenge, n_challenges=args.n_games)
    elif args.mode == "ladder":
        await bot.ladder(args.n_games)
    else:
        await bot.accept_challenges(None, args.n_games)

    print(f"finished: {bot.n_won_battles}W / {bot.n_lost_battles}L / "
          f"{bot.n_tied_battles}T", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
