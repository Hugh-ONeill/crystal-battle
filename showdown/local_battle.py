# local battle driver: MCTS vs SmartAgent using poke-engine
# no Showdown server needed -- runs entirely in Python/Rust
#
# Usage:
#   .venv/bin/python showdown/local_battle.py --games 20 --search-ms 500

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.poke_engine_player import PokeEngineTranslator
from showdown.chaos_stats import ChaosStats, RevealedMon
from showdown.name_mapping import _normalize
from showdown.features_v2 import get_move_props, type_effectiveness


# ============================================================
# TEAM BUILDING (from sample team strings -> poke-engine State)
# ============================================================

def parse_showdown_team(team_str: str) -> list[dict]:
    """Parse a Showdown paste into a list of mon dicts."""
    mons = []
    current = None
    for line in team_str.strip().split("\n"):
        line = line.strip()
        if not line:
            if current:
                mons.append(current)
            current = None
            continue
        if current is None:
            current = {"moves": [], "item": "leftovers", "ivs": {}}
            if "@" in line:
                parts = line.split("@")
                current["species"] = _normalize(parts[0].split("(")[0].strip())
                current["item"] = _normalize(parts[1].strip())
            else:
                current["species"] = _normalize(line.split("(")[0].strip())
        elif line.startswith("- "):
            move = _normalize(line[2:].split("[")[0].strip())
            # handle Hidden Power [Type]
            if "hiddenpower" in _normalize(line[2:]):
                move = _normalize(line[2:].replace("[", "").replace("]", "").strip())
            current["moves"].append(move)
        elif line.startswith("IVs:"):
            pass  # skip IV line for now
    if current:
        mons.append(current)
    return mons


def build_pe_state(team1_str: str, team2_str: str) -> pe.State:
    """Build a poke-engine State from two Showdown team pastes."""
    from poke_env.data.gen_data import GenData
    gd = GenData.from_gen(2)
    pokedex = gd.pokedex

    def build_side(team_str):
        mons_data = parse_showdown_team(team_str)
        pokemon = []
        for md in mons_data:
            bs = pokedex.get(md["species"], {}).get("baseStats", {})
            def calc(base, is_hp=False):
                core = ((base + 15) * 2 + 64) * 100 // 100
                return core + 110 if is_hp else core + 5

            types_list = pokedex.get(md["species"], {}).get("types", ["Normal"])
            types = tuple(t.lower() for t in types_list)
            if len(types) < 2:
                types = (types[0], "typeless")

            moves = [pe.Move(id=m, pp=16) for m in md["moves"][:4]]
            while len(moves) < 4:
                moves.append(pe.Move(id="splash", pp=1))

            pokemon.append(pe.Pokemon(
                id=md["species"], level=100,
                hp=calc(bs.get("hp", 80), is_hp=True),
                maxhp=calc(bs.get("hp", 80), is_hp=True),
                attack=calc(bs.get("atk", 80)),
                defense=calc(bs.get("def", 80)),
                special_attack=calc(bs.get("spa", 80)),
                special_defense=calc(bs.get("spd", 80)),
                speed=calc(bs.get("spe", 80)),
                types=types, ability="noability",
                item=md.get("item", "leftovers"),
                status="none", moves=moves,
            ))
        while len(pokemon) < 6:
            pokemon.append(pe.Pokemon.create_fainted())
        return pe.Side(pokemon=pokemon[:6])

    side1 = build_side(team1_str)
    side2 = build_side(team2_str)
    return pe.State(
        side_one=side1, side_two=side2,
        weather=pe.Weather.NONE, weather_turns_remaining=0,
        terrain=pe.Terrain.NONE, terrain_turns_remaining=0,
        trick_room=False, trick_room_turns_remaining=0,
        team_preview=False,
    )


# ============================================================
# SMART AGENT (operates on poke-engine state)
# ============================================================

def _pokemon_types_upper(p) -> tuple:
    """Return (type1, type2) upper-cased; type2 may be 'TYPELESS'."""
    t1 = (p.types[0] if len(p.types) >= 1 else "normal").upper()
    t2 = (p.types[1] if len(p.types) >= 2 else "typeless").upper()
    return t1, t2


def _move_potential(move_id: str, user_types: tuple, opp_types: tuple) -> float:
    """Approximate a move's offensive value vs an opponent.

    Returns 0 for status moves and 0-effectiveness matchups (immunities).
    Uses simple power × effectiveness × STAB; ignores stat differences.
    """
    if move_id == "none":
        return 0.0
    move_type, power, accuracy, is_status, _ = get_move_props(move_id)
    if is_status or power <= 0:
        return 0.0
    eff = type_effectiveness(move_type, opp_types[0], opp_types[1])
    if eff == 0.0:
        return 0.0
    stab = 1.5 if move_type in user_types else 1.0
    return power * eff * stab * (accuracy / 100.0)


def _best_move_potential(p, opp_types: tuple) -> float:
    """Best offensive potential among a Pokemon's known moves vs opp_types."""
    user_types = _pokemon_types_upper(p)
    best = 0.0
    for move in p.moves:
        if move.pp <= 0:
            continue
        score = _move_potential(move.id, user_types, opp_types)
        if score > best:
            best = score
    return best


def _pick_switch_target(my_side, opp_active) -> str:
    """Pick a non-active, alive Pokemon that can hit opp_active. Empty string if none viable."""
    active_idx = int(my_side.active_index)
    opp_types = _pokemon_types_upper(opp_active)

    best_id = ""
    best_score = -1.0
    for i, p in enumerate(my_side.pokemon):
        if i == active_idx or p.hp <= 0:
            continue
        offensive = _best_move_potential(p, opp_types)
        # incoming threat: opp's best move vs us
        my_types = _pokemon_types_upper(p)
        incoming = _best_move_potential(opp_active, my_types)
        # bias toward (a) can hit them (b) lower incoming threat (c) more HP
        hp_frac = p.hp / max(p.maxhp, 1)
        score = offensive - 0.5 * incoming + 50.0 * hp_frac
        # heavy penalty for total-immunity targets so we never switch into one when better exists
        if offensive == 0.0:
            score -= 10000.0
        if score > best_score:
            best_score = score
            best_id = p.id.lower()
    return best_id


def smart_pick_move(state: pe.State, side: str = "s2") -> str:
    """Heuristic: pick highest-damage move; switch out of hopeless matchups.

    Returns a string in the format MoveChoice::from_string accepts:
    pokemon species id for switches, move id for moves, "none" if no action.
    """
    if side == "s2":
        my_side = state.side_two
        opp_side = state.side_one
        target_tag = "SideOne"
        dmg_call = lambda mid: pe.generate_instructions(state, "none", mid)
    else:
        my_side = state.side_one
        opp_side = state.side_two
        target_tag = "SideTwo"
        dmg_call = lambda mid: pe.generate_instructions(state, mid, "none")

    active = my_side.pokemon[int(my_side.active_index)]
    opp_active = opp_side.pokemon[int(opp_side.active_index)]

    # forced switch: pick best switch target (or healthiest fallback)
    if active.hp <= 0:
        target = _pick_switch_target(my_side, opp_active)
        if target:
            return target
        # no viable damaging switch -- pick healthiest
        best_idx = -1
        best_hp = -1
        for i, p in enumerate(my_side.pokemon):
            if p.hp > 0 and i != int(my_side.active_index) and p.hp > best_hp:
                best_idx = i
                best_hp = p.hp
        return my_side.pokemon[best_idx].id.lower() if best_idx >= 0 else "none"

    # score each move by expected damage
    best_move = None
    best_score = -1.0
    opp_types = _pokemon_types_upper(opp_active)
    user_types = _pokemon_types_upper(active)

    for move in active.moves:
        if move.id == "none" or move.pp <= 0:
            continue
        # quick reject: if move is 0x effective by type, skip the expensive sim
        if _move_potential(move.id, user_types, opp_types) == 0.0:
            # but still consider status moves at 0 -- they tie -- handled by the loop
            pass
        try:
            insts = dmg_call(move.id)
            total_dmg = 0.0
            for inst in insts:
                for instr_item in inst.instruction_list:
                    s = str(instr_item)
                    if "Damage" in s and target_tag in s:
                        parts = s.split(":")
                        if len(parts) >= 2:
                            try:
                                dmg = int(parts[-1].strip())
                                total_dmg += dmg * inst.percentage / 100.0
                            except ValueError:
                                pass
            if total_dmg > best_score:
                best_score = total_dmg
                best_move = move.id
        except Exception:
            continue

    # if no move does damage (e.g. all immune / all status), try to switch out
    if best_score <= 0.0:
        target = _pick_switch_target(my_side, opp_active)
        if target:
            return target

    return best_move if best_move is not None else "none"


# ============================================================
# GAME DRIVER
# ============================================================

def _find_move_index(side_results, move_choice: str) -> int:
    """Find the index of a move_choice in a list of MctsSideResult; -1 if not found.
    Tries exact match, then 'switch <name>' form for switch targets."""
    for i, r in enumerate(side_results):
        if r.move_choice == move_choice:
            return i
    switch_form = f"switch {move_choice}"
    for i, r in enumerate(side_results):
        if r.move_choice == switch_form:
            return i
    return -1


def play_game(team1_str: str, team2_str: str, search_ms: int = 500,
              n_samples: int = 1, verbose: bool = False,
              policy_net=None, value_net=None, alpha: float = 1.0,
              max_turns: int = 250, p2_mode: str = "mcts",
              tree_reuse: bool = False) -> int:
    """Play one game: MCTS (P1) vs P2 (mode-controlled).

    policy_net: optional PolicyNet instance (Python). When set, used for PUCT priors.
    value_net: optional pe.ValueNet instance. When set, used for MCTS leaf eval.
    p2_mode: "mcts" (50ms MCTS, default) or "smart" (smart_pick_move heuristic).
    tree_reuse: if True, persist the MCTS tree across turns (plain MCTS only).

    Returns: 1 if P1 wins, 2 if P2 wins, 0 if draw/timeout.
    """
    state = build_pe_state(team1_str, team2_str)
    # tree_reuse only supported for plain MCTS (no priors / no value net) for now
    use_tree = tree_reuse and policy_net is None and value_net is None
    tree = None  # pe.MctsTree | None

    for turn in range(max_turns):
        # check if battle is over
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            return 2
        if s2_alive == 0:
            return 1

        # check for battle_is_over from poke-engine's perspective
        s_str = state.to_string()

        # P1: MCTS, dispatching by which models are loaded
        try:
            s1_priors = None
            s2_priors = None
            if policy_net is not None:
                if policy_net.state_dim == 609:
                    from showdown.features_v2 import parse_state_v2
                    features = parse_state_v2(s_str)
                else:
                    from showdown.policy_train import parse_state_string
                    features = parse_state_string(s_str)
                probs = policy_net.predict(features)[0]
                n = len(probs)
                s1_priors = probs.tolist()
                s2_priors = [1.0 / n] * n

            if value_net is not None:
                result = pe.monte_carlo_tree_search_with_value(
                    state, value_net, search_ms, s1_priors, s2_priors, alpha,
                )
                p1_move = max(result.side_one, key=lambda x: x.visits).move_choice
            elif s1_priors is not None:
                result = pe.monte_carlo_tree_search_with_priors(
                    state, s1_priors, s2_priors, search_ms,
                )
                p1_move = max(result.side_one, key=lambda x: x.visits).move_choice
            elif use_tree:
                if tree is None:
                    tree = pe.MctsTree(state, search_ms)
                else:
                    tree.search(state, search_ms)
                # raw result has .s1/.s2 (not .side_one/.side_two)
                tree_res = tree.result(state)
                p1_idx = max(range(len(tree_res.s1)), key=lambda i: tree_res.s1[i].visits)
                p1_move = tree_res.s1[p1_idx].move_choice
            else:
                result = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
                p1_move = max(result.side_one, key=lambda x: x.visits).move_choice
        except Exception as e:
            if verbose:
                print(f"  MCTS error: {e}")
            break

        # P2: either fast MCTS or hand-written damage heuristic
        try:
            if p2_mode == "smart":
                p2_move = smart_pick_move(state, side="s2")
            else:
                p2_result = pe.monte_carlo_tree_search(state, duration_ms=50)
                p2_move = max(p2_result.side_two, key=lambda x: x.visits).move_choice
        except Exception as e:
            if verbose:
                print(f"  P2 error: {e}")
            break

        if verbose:
            p1_active = state.side_one.pokemon[int(state.side_one.active_index)]
            p2_active = state.side_two.pokemon[int(state.side_two.active_index)]
            print(f"  T{turn+1}: {p1_active.id}({p1_active.hp}) {p1_move} "
                  f"vs {p2_active.id}({p2_active.hp}) {p2_move}")

        # strip "switch " prefix
        if p1_move.startswith("switch "):
            p1_move = p1_move[7:]
        if p2_move.startswith("switch "):
            p2_move = p2_move[7:]

        # "No Move" means that side has nothing to do this turn (e.g. forced
        # switch on the other side). MoveChoice::from_string parses "none"
        # back to MoveChoice::None, so map here. Only abort if BOTH sides
        # have no move -- the engine treats that as game-over.
        if p1_move == "No Move" and p2_move == "No Move":
            if verbose:
                print(f"  Both No Move -- game over")
            break
        if p1_move == "No Move":
            p1_move = "none"
        if p2_move == "No Move":
            p2_move = "none"

        # resolve turn
        try:
            instructions = pe.generate_instructions(state, p1_move, p2_move)
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            break

        if not instructions:
            break

        # pick outcome weighted by probability
        roll = random.random() * 100
        cumulative = 0
        chosen = instructions[0]
        for inst in instructions:
            cumulative += inst.percentage
            if roll <= cumulative:
                chosen = inst
                break

        new_state = state.apply_instructions(chosen)

        # tree reuse: rebase the search tree to the realized child
        if use_tree and tree is not None:
            # find s2 index in tree's option list (smart returns move id or species)
            s2_idx = _find_move_index(tree_res.s2,
                                      p2_move if p2_move != "none" else "none")
            if s2_idx >= 0:
                ok = tree.rebase(p1_idx, s2_idx, chosen, new_state)
                if not ok:
                    tree = None  # rebuild next turn
            else:
                tree = None  # couldn't locate p2's move in tree options

        state = new_state

    # timeout = draw
    return 0


def main():
    parser = argparse.ArgumentParser(description="Local MCTS vs Smart battles")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--team1", type=int, default=0, help="Sample team index for MCTS")
    parser.add_argument("--team2", type=int, default=0, help="Sample team index for Smart")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--policy-net-path", type=str, default=None,
                        help="path to a trained policy net .pt (PUCT priors)")
    parser.add_argument("--value-net-path", type=str, default=None,
                        help="path to a trained value net .onnx (MCTS leaf eval)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="leaf-eval mix: alpha*value_net + (1-alpha)*heuristic")
    parser.add_argument("--max-turns", type=int, default=250,
                        help="hard turn cap; games over this become draws")
    parser.add_argument("--p2", choices=["mcts", "smart"], default="mcts",
                        help="P2 opponent: 'mcts' (50ms MCTS) or 'smart' "
                             "(damage-maximizing heuristic).")
    parser.add_argument("--tree-reuse", action="store_true",
                        help="Persist MCTS tree across turns (plain MCTS only).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Per-game seed base; game i uses random.seed(seed+i). "
                             "Use the same seed across A/B runs for paired comparison.")
    args = parser.parse_args()

    from showdown.sample_teams import SAMPLE_TEAMS
    team1 = SAMPLE_TEAMS[args.team1]
    team2 = SAMPLE_TEAMS[args.team2]

    policy_net = None
    if args.policy_net_path:
        import torch
        from showdown.policy_train import PolicyNet
        ckpt = torch.load(args.policy_net_path, map_location="cpu", weights_only=True)
        policy_net = PolicyNet(state_dim=ckpt["state_dim"], hidden=ckpt["hidden"])
        policy_net.load_state_dict(ckpt["model"])
        policy_net.eval()
        policy_net.state_dim = ckpt["state_dim"]
        print(f"Loaded policy net from {args.policy_net_path}")

    value_net = None
    if args.value_net_path:
        value_net = pe.ValueNet(args.value_net_path)
        print(f"Loaded value net from {args.value_net_path}")

    t0 = time.time()
    wins = losses = draws = 0
    for i in range(args.games):
        if args.seed is not None:
            random.seed(args.seed + i)
        result = play_game(team1, team2, search_ms=args.search_ms,
                           n_samples=args.n_samples, verbose=args.verbose,
                           policy_net=policy_net, value_net=value_net,
                           alpha=args.alpha, max_turns=args.max_turns,
                           p2_mode=args.p2, tree_reuse=args.tree_reuse)
        if result == 1:
            wins += 1
            print("W", end="", flush=True)
        elif result == 2:
            losses += 1
            print("L", end="", flush=True)
        else:
            draws += 1
            print("D", end="", flush=True)

    elapsed = time.time() - t0
    total = wins + losses
    pct = wins / total * 100 if total > 0 else 0
    print(f"\n\nMCTS vs Smart: {wins}W {losses}L {draws}D "
          f"({pct:.0f}%) in {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")


if __name__ == "__main__":
    main()
