"""
Per-side move-selection heuristics layered on top of MCTS picks.

`recovery_override(side, mcts_results, prev_hp, prev_active_id)`:
  When the active mon is at < 50% HP, has a 50% recovery move available in
  the MCTS option set, and the damage it took last turn would have been
  out-healed by that recovery, prefer the recovery over MCTS's pick.

This addresses an engine bias surfaced in monotype traces (Avalugg burning
to death while picking Iron Defense / Body Press over Recover) — see
[[project-monotype-bench-interpretation]].
"""

from __future__ import annotations


# Heal-half-max-HP moves. Weather-conditional ones (Moonlight/Morning Sun/
# Synthesis) heal 50% in clear weather, 67% in sun, 25% in rain/snow/sand —
# treat them as 50% for the threshold check (conservative).
HEAL_HALF_MAX = frozenset({
    "recover", "roost", "softboiled", "slackoff", "milkdrink", "shoreup",
    "moonlight", "morningsun", "synthesis",
})
# Quarter-heal moves (Life Dew, Jungle Healing partial). Lower priority.
HEAL_QUARTER_MAX = frozenset({"lifedew"})

# Self-buff moves: only affect the user's own stats. Wasted if the user
# dies the same/next turn. These are the only MCTS picks we override —
# trying to override attacks or status-on-opponent moves backfired in v6
# bench (recovery mons lost tempo).
SELF_BUFF_MOVES = frozenset({
    "swordsdance", "nastyplot", "calmmind", "bulkup", "irondefense",
    "amnesia", "agility", "rockpolish", "shellsmash", "shiftgear",
    "dragondance", "quiverdance", "noretreat", "geomancy",
    "cosmicpower", "stockpile", "acidarmor", "barrier", "defendorder",
    "tailglow", "victorydance", "coil", "growth", "workup", "howl",
    "meditate", "sharpen", "harden", "withdraw", "curse",
})


def recovery_override(
    side,
    mcts_results,
    prev_hp: int | None,
    prev_active_id: str | None,
    *,
    hp_threshold: float = 0.50,
):
    """Return a move_choice string to override MCTS, or None.

    Args:
        side:               poke_engine Side object (state.side_one or two)
        mcts_results:       list of SideMoveResult (visits per move_choice)
        prev_hp:            active mon's HP at the start of the previous turn
                            (None if first turn or after a switch)
        prev_active_id:     active mon's id at the start of the previous turn
                            (used to detect that a switch happened)
        hp_threshold:       fire only when current HP fraction is below this
    """
    if side.active_index is None:
        return None
    try:
        idx = int(side.active_index)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(side.pokemon):
        return None
    active = side.pokemon[idx]
    if active.maxhp <= 0 or active.hp <= 0:
        return None
    hp_frac = active.hp / active.maxhp
    if hp_frac >= hp_threshold:
        return None

    # Find which moves are available (excluding tera variants).
    available: dict[str, int] = {}
    for r in mcts_results:
        mc = r.move_choice
        if mc.endswith("-tera"):
            continue
        available[mc] = r.visits
    if not available:
        return None

    # Don't override unless MCTS's top pick is a self-buff. Trying to
    # override attacks / status / hazards / etc. broke v6 bench results
    # for recovery-mon teams (Blissey, Toxapex, Slowbro, etc.) — they lost
    # tempo when MCTS's chip turn was forcibly replaced.
    top_move = max(available.items(), key=lambda kv: kv[1])[0]
    if top_move not in SELF_BUFF_MOVES:
        return None

    recovery_options: list[tuple[str, float]] = []
    for mv in HEAL_HALF_MAX:
        if mv in available:
            recovery_options.append((mv, 0.5))
    for mv in HEAL_QUARTER_MAX:
        if mv in available:
            recovery_options.append((mv, 0.25))
    if not recovery_options:
        return None

    # Damage estimate from the previous turn. Reset to "unknown" if the
    # active mon changed between turns (switch) — we don't have history
    # for the new mon's most-recent hit.
    same_active = (prev_active_id is not None
                   and prev_active_id == active.id
                   and prev_hp is not None)
    if not same_active:
        # No prior damage signal — fire only if HP is already quite low so
        # recovery is unambiguously the safe play.
        if hp_frac < 0.30:
            best = max(recovery_options, key=lambda x: x[1])
            return best[0]
        return None

    damage_taken = max(0, prev_hp - active.hp)

    # Prefer the highest-heal recovery move whose heal beats damage taken.
    # If damage_taken is 0 (e.g. last turn was a status setup with no hit),
    # the heuristic still fires since heal will always exceed 0.
    recovery_options.sort(key=lambda x: -x[1])
    for mv, heal_frac in recovery_options:
        heal_amount = int(active.maxhp * heal_frac)
        if damage_taken < heal_amount:
            return mv
    return None
