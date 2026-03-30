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

def smart_pick_move(state: pe.State, side: str = "s2") -> str:
    """Simple heuristic: pick highest-damage move for side.

    Uses generate_instructions to estimate damage for each move option.
    """
    # get available moves by trying generate_instructions
    # use a dummy move for the other side
    if side == "s2":
        my_side = state.side_two
        opp_side = state.side_one
    else:
        my_side = state.side_one
        opp_side = state.side_two

    active = my_side.pokemon[int(my_side.active_index)]
    if active.hp <= 0:
        # need to switch -- pick healthiest
        for i, p in enumerate(my_side.pokemon):
            if p.hp > 0 and i != int(my_side.active_index):
                return f"switch {i}"
        return active.moves[0].id

    # score each move by expected damage
    best_move = active.moves[0].id
    best_score = -999

    for move in active.moves:
        if move.id == "none" or move.pp <= 0:
            continue
        try:
            if side == "s2":
                insts = pe.generate_instructions(state, "splash", move.id)
            else:
                insts = pe.generate_instructions(state, move.id, "splash")

            # estimate damage from the weighted instructions
            total_dmg = 0
            for inst in insts:
                for instr_item in inst.instruction_list:
                    s = str(instr_item)
                    # parse "Damage SideOne: X" or "Damage SideTwo: X"
                    if "Damage" in s:
                        parts = s.split(":")
                        if len(parts) >= 2:
                            dmg = int(parts[-1].strip())
                            # check if damage is to the opponent
                            if (side == "s2" and "SideOne" in s) or \
                               (side == "s1" and "SideTwo" in s):
                                total_dmg += dmg * inst.percentage / 100
            if total_dmg > best_score:
                best_score = total_dmg
                best_move = move.id
        except Exception:
            continue

    return best_move


# ============================================================
# GAME DRIVER
# ============================================================

def play_game(team1_str: str, team2_str: str, search_ms: int = 500,
              n_samples: int = 1, verbose: bool = False) -> int:
    """Play one game: MCTS (P1) vs Smart heuristic (P2).

    Returns: 1 if P1 wins, 2 if P2 wins, 0 if draw/timeout.
    """
    state = build_pe_state(team1_str, team2_str)

    for turn in range(100):
        # check if battle is over
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            return 2
        if s2_alive == 0:
            return 1

        # check for battle_is_over from poke-engine's perspective
        s_str = state.to_string()

        # P1: MCTS
        try:
            result = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
            p1_move = max(result.side_one, key=lambda x: x.visits).move_choice
        except Exception as e:
            if verbose:
                print(f"  MCTS error: {e}")
            break

        # P2: use MCTS too but with very short time (acts as smart heuristic)
        try:
            # flip sides: P2's perspective
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

        if p1_move == "No Move" or p2_move == "No Move":
            if verbose:
                print(f"  No Move -- game over")
            break

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

        state = state.apply_instructions(chosen)

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
    args = parser.parse_args()

    from showdown.sample_teams import SAMPLE_TEAMS
    team1 = SAMPLE_TEAMS[args.team1]
    team2 = SAMPLE_TEAMS[args.team2]

    t0 = time.time()
    wins = losses = draws = 0
    for i in range(args.games):
        result = play_game(team1, team2, search_ms=args.search_ms,
                           n_samples=args.n_samples, verbose=args.verbose)
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
