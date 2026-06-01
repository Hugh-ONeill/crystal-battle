"""
Endgame solver — exhaustive minimax with chance branches for low-alive-count
states where MCTS at 200-500 ms gives noisy answers but the true state space
is tractable.

V1 scope: 1v1 endgames (both sides exactly 1 alive mon). Each side has only
the active mon's moves as legal actions (no switch options when only 1 mon
remains). The search recurses through `generate_instructions` chance
branches until faint, with memoization on `state.to_string()`.

Trigger: `is_solvable_endgame(state)` — currently 1v1 only.
Return:   `solve_endgame(state)` → (p1_move_id, p2_move_id, expected_value_for_p1)

Expected value is in [-1, +1]: +1 = P1 sure win, -1 = P2 sure win,
0 = draw (or terminal hit max-depth fallback).
"""

from __future__ import annotations

from functools import lru_cache

import poke_engine as pe


def _alive(side) -> int:
    return sum(1 for p in side.pokemon if p.hp > 0)


def is_solvable_endgame(state, max_total_alive: int = 3) -> bool:
    """Gate: total alive across both sides <= threshold. 1v1 / 2v1 / 1v2
    qualify by default; 2v2 (4 alive) is sometimes feasible but blows up
    the branching factor."""
    return _alive(state.side_one) + _alive(state.side_two) <= max_total_alive


def _legal_moves(side) -> list[str]:
    """Return non-disabled, non-tera move ids for `side`'s active mon."""
    active = side.pokemon[int(side.active_index)]
    out = []
    for m in active.moves:
        if m.disabled:
            continue
        mid = (m.id or "").lower()
        if not mid or mid == "none":
            continue
        out.append(mid)
    return out


def _legal_switches(side) -> list[str]:
    """Return species ids of switch-eligible benchmons (alive, not active)."""
    out = []
    active_idx = int(side.active_index)
    for i, p in enumerate(side.pokemon):
        if i == active_idx:
            continue
        if p.hp <= 0:
            continue
        out.append((p.id or "").lower())
    return out


def _legal_actions(side) -> list[str]:
    """All legal actions for a side: moves + 'switch <species>' targets.
    The engine expects switch actions as bare species ids (no 'switch ' prefix)
    when passed to generate_instructions, but we tag them with 'switch ' here
    for clarity and strip it at call time."""
    moves = _legal_moves(side)
    switches = [f"switch {sp}" for sp in _legal_switches(side)]
    return moves + switches


def _strip_switch(a: str) -> str:
    return a[7:] if a.startswith("switch ") else a


def _terminal_value(state) -> float | None:
    """Return +1/-1/0 if terminal, else None."""
    s1, s2 = _alive(state.side_one), _alive(state.side_two)
    if s1 > 0 and s2 > 0:
        return None
    if s1 == 0 and s2 == 0:
        return 0.0
    return 1.0 if s2 == 0 else -1.0


def solve_endgame(
    state,
    max_depth: int = 30,
    memo: dict | None = None,
) -> tuple[str | None, str | None, float]:
    """Solve a 1v1 endgame. Returns (p1_action, p2_action, value) where
    value is in [-1, +1] from P1's perspective. Memo dict is created
    fresh if not supplied; pass one to reuse across calls.
    """
    if memo is None:
        memo = {}
    return _solve_recursive(state, depth=0, max_depth=max_depth, memo=memo,
                            want_actions=True)


def _value_only(state, depth: int, max_depth: int, memo: dict) -> float:
    """Recursion helper — value only, no action returned (faster path)."""
    t = _terminal_value(state)
    if t is not None:
        return t
    if depth >= max_depth:
        # Heuristic: engine static eval clipped to [-1, +1] for value
        # consistency. ~100 maps to ~0.5 typical advantage.
        return max(-1.0, min(1.0, pe.evaluate(state) / 200.0))
    key = state.to_string()
    if key in memo:
        return memo[key]

    p1_moves = _legal_actions(state.side_one)
    p2_moves = _legal_actions(state.side_two)
    if not p1_moves or not p2_moves:
        # No options — degenerate, just use eval
        v = max(-1.0, min(1.0, pe.evaluate(state) / 200.0))
        memo[key] = v
        return v

    # Build value matrix: matrix[(m1, m2)] = expected value
    matrix: dict[tuple[str, str], float] = {}
    for m1 in p1_moves:
        for m2 in p2_moves:
            try:
                branches = pe.generate_instructions(
                    state, _strip_switch(m1), _strip_switch(m2))
            except Exception:
                continue
            if not branches:
                continue
            ev = 0.0
            for br in branches:
                ns = state.apply_instructions(br)
                ev += (br.percentage / 100.0) * _value_only(
                    ns, depth + 1, max_depth, memo)
            matrix[(m1, m2)] = ev

    if not matrix:
        v = 0.0
        memo[key] = v
        return v

    # Maximin from P1's perspective. (Pure-strategy approximation of the
    # Nash equilibrium of the simultaneous-move 4x4 matrix game; LP-solved
    # mixed strategy would be more accurate but adds dep + complexity.)
    p1_minvals: dict[str, float] = {}
    for m1 in p1_moves:
        worst = float("inf")
        for m2 in p2_moves:
            v = matrix.get((m1, m2))
            if v is None:
                continue
            if v < worst:
                worst = v
        if worst != float("inf"):
            p1_minvals[m1] = worst
    if not p1_minvals:
        v = 0.0
        memo[key] = v
        return v
    best_v = max(p1_minvals.values())
    memo[key] = best_v
    return best_v


def _solve_recursive(state, depth, max_depth, memo, want_actions: bool):
    """Action-aware top-level — only called once per query."""
    t = _terminal_value(state)
    if t is not None:
        return (None, None, t)

    p1_moves = _legal_actions(state.side_one)
    p2_moves = _legal_actions(state.side_two)
    if not p1_moves or not p2_moves:
        return (None, None, 0.0)

    matrix: dict[tuple[str, str], float] = {}
    for m1 in p1_moves:
        for m2 in p2_moves:
            try:
                branches = pe.generate_instructions(
                    state, _strip_switch(m1), _strip_switch(m2))
            except Exception:
                continue
            if not branches:
                continue
            ev = 0.0
            for br in branches:
                ns = state.apply_instructions(br)
                ev += (br.percentage / 100.0) * _value_only(
                    ns, depth + 1, max_depth, memo)
            matrix[(m1, m2)] = ev

    if not matrix:
        return (None, None, 0.0)

    # P1 maximin
    best_p1 = None
    best_p1_v = -float("inf")
    for m1 in p1_moves:
        worst = float("inf")
        for m2 in p2_moves:
            v = matrix.get((m1, m2))
            if v is None:
                continue
            worst = min(worst, v)
        if worst == float("inf"):
            continue
        if worst > best_p1_v:
            best_p1_v = worst
            best_p1 = m1

    # P2 minimax (P2 wants P1's value low)
    best_p2 = None
    best_p2_v = float("inf")
    for m2 in p2_moves:
        worst = -float("inf")  # worst for P2 = best for P1
        for m1 in p1_moves:
            v = matrix.get((m1, m2))
            if v is None:
                continue
            worst = max(worst, v)
        if worst == -float("inf"):
            continue
        if worst < best_p2_v:
            best_p2_v = worst
            best_p2 = m2

    return (best_p1, best_p2, best_p1_v)
