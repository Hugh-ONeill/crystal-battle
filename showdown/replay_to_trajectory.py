#!/usr/bin/env python3
"""Engine-step a parsed gen9 replay into a sequence of (state, label) training tuples.

Step 4 of the gen9 value-net pipeline: given a Showdown replay JSON and a
ChaosStats instance, produce a list of `(state_string, label, turns_remaining)`
tuples by replaying the recorded actions through poke-engine.

The state we record after each apply is the engine's view, which may diverge
slightly from the real replay (different damage rolls, etc.). That's OK for
value-net training — the labels are correct; the states are plausibly close.
"""

from __future__ import annotations

import random

import poke_engine as pe

from showdown.chaos_stats import ChaosStats
from showdown.local_battle import build_pe_state_gen9
from showdown.name_mapping import _normalize
from showdown.replay_parse_gen9 import parse_replay
from showdown.team_reconstruction import reconstruct_team


def _needs_switch(side) -> bool:
    """Engine's force_switch flag isn't always set after a faint mid-pivot;
    the reliable signal is that the active mon has 0 hp."""
    if side.force_switch:
        return True
    active = side.pokemon[int(side.active_index)]
    return active.hp <= 0


def _pick_alive_teammate(side) -> str | None:
    """Fallback switch target when our engine sim faints a mon the replay didn't,
    so no recorded pivot exists. Pick the first alive non-active teammate."""
    active_idx = int(side.active_index)
    for i, p in enumerate(side.pokemon):
        if i == active_idx:
            continue
        if p.hp > 0:
            return _normalize(p.id)
    return None


def _has_queued_move(side) -> bool:
    """True if the side has a saved move pending resolution (e.g. Foul Play
    queued behind an opponent's U-turn pivot). When True, the next pivot
    iteration must pass that move so the engine can resolve it; passing 'none'
    drops the queued move and diverges state from the replay."""
    saved = side.switch_out_move_second_saved_move
    return saved is not None and str(saved).upper() != "NONE"


def _action_to_engine_str(action: dict | None, terad: bool) -> str | None:
    """Translate a parser action dict to the string poke-engine expects.

    - Move: "<moveid>" or "<moveid>-tera" if this side terastallized this turn.
    - Switch: "<species>" (engine accepts the bare species id; bench scripts
      strip the "switch " prefix before calling generate_instructions).
    """
    if action is None:
        return None
    name = _normalize(action.get("name", ""))
    if not name:
        return None
    atype = action.get("type")
    if atype == "move":
        return f"{name}-tera" if terad else name
    if atype == "switch":
        return name
    return None


def replay_to_trajectory(replay_json: dict,
                         chaos: ChaosStats) -> list[tuple[str, float, int]]:
    """Produce (state_str, label, turns_remaining) tuples for a single replay.

    Returns [] if the replay is aborted, has no winner, or both teams cannot be
    reconstructed. On engine error mid-game, returns whatever was recorded up
    to that point.
    """
    traj = parse_replay(replay_json)
    if traj.aborted or traj.winner is None:
        return []

    if traj.winner == "p1":
        label = 1.0
    elif traj.winner == "p2":
        label = 0.0
    else:  # "tie"
        label = 0.5

    try:
        team1_str = reconstruct_team(traj.p1_team, chaos, lead_species=traj.p1_lead)
        team2_str = reconstruct_team(traj.p2_team, chaos, lead_species=traj.p2_lead)
        state = build_pe_state_gen9(team1_str, team2_str)
    except Exception:
        return []

    states_recorded: list[str] = []
    prev_str: str | None = None
    stuck = 0

    def _step(state, a1: str, a2: str):
        """Apply one generate_instructions/apply pair, sampling by percentage."""
        instructions = pe.generate_instructions(state, a1, a2)
        if not instructions:
            return None
        roll = random.random() * 100
        cum = 0.0
        chosen = instructions[0]
        for inst in instructions:
            cum += inst.percentage
            if roll <= cum:
                chosen = inst
                break
        return state.apply_instructions(chosen)

    for turn in traj.turns:
        s1 = _action_to_engine_str(turn.get("p1_action"), turn.get("p1_terad", False)) or "none"
        s2 = _action_to_engine_str(turn.get("p2_action"), turn.get("p2_terad", False)) or "none"
        if s1 == "none" and s2 == "none":
            break

        try:
            new_state = _step(state, s1, s2)
        except Exception:
            break
        if new_state is None:
            break
        state = new_state

        # Resolve any pending switches. Two scenarios:
        #   - U-turn pivot pause: side_one used U-turn, side_two's move queued.
        #     Engine waits for pivot target; pivot step passes pivot for s1 and
        #     side_two's *queued* move for s2 so the queued resolution fires.
        #   - Post-faint switch: a side fainted mid-turn. Other side's move
        #     already fired, no queued move. Pass 'none' for the non-switching
        #     side so the engine doesn't fire a move that already happened.
        # Distinguisher: switch_out_move_second_saved_move (engine's queued-
        # move slot). If set, pass remaining_action; if NONE, pass 'none'.
        remaining_s1, remaining_s2 = s1, s2
        used_p1_pivot = used_p2_pivot = False
        aborted_turn = False
        for _ in range(3):
            if not (_needs_switch(state.side_one) or _needs_switch(state.side_two)):
                break
            if _needs_switch(state.side_one):
                pivot1 = turn.get("p1_pivot") or _pick_alive_teammate(state.side_one)
                if not pivot1:
                    aborted_turn = True
                    break
                ns1 = _normalize(pivot1) if turn.get("p1_pivot") else pivot1
                used_p1_pivot = True
            else:
                ns1 = remaining_s1 if _has_queued_move(state.side_one) else "none"
            if _needs_switch(state.side_two):
                pivot2 = turn.get("p2_pivot") or _pick_alive_teammate(state.side_two)
                if not pivot2:
                    aborted_turn = True
                    break
                ns2 = _normalize(pivot2) if turn.get("p2_pivot") else pivot2
                used_p2_pivot = True
            else:
                ns2 = remaining_s2 if _has_queued_move(state.side_two) else "none"
            try:
                new_state = _step(state, ns1, ns2)
            except Exception:
                aborted_turn = True
                break
            if new_state is None:
                aborted_turn = True
                break
            state = new_state
            remaining_s1 = "none"
            remaining_s2 = "none"
        if aborted_turn:
            break

        # Replay-alignment: the replay logged a pivot that our engine never
        # asked for (typical when our damage rolls didn't faint the mon the
        # replay fainted). Synthesize a one-sided switch so the active mon
        # matches what the next turn's actions assume — otherwise we'd hit
        # "Invalid move" as soon as the replay's new active uses a move the
        # diverged active doesn't have. State diverges further on HP / status,
        # which is fine for value-net training per the plan's gotcha #5.
        # If the engine rejects the synthesized switch (target fainted in our
        # sim, name mismatch, etc.), leave the misalignment and let the next
        # turn break naturally — don't pretend we resolved it.
        for which, used, pivot_key in (
            (1, used_p1_pivot, "p1_pivot"),
            (2, used_p2_pivot, "p2_pivot"),
        ):
            if used:
                continue
            pivot = turn.get(pivot_key)
            if not pivot:
                continue
            target = _normalize(pivot)
            sa1, sa2 = (target, "none") if which == 1 else ("none", target)
            try:
                new_state = _step(state, sa1, sa2)
            except Exception:
                new_state = None
            if new_state is not None:
                state = new_state

        cur_str = state.to_string()
        if cur_str == prev_str:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0
        prev_str = cur_str
        states_recorded.append(cur_str)

    n = len(states_recorded)
    return [(s, label, n - i - 1) for i, s in enumerate(states_recorded)]
