"""
PUCT-style MCTS wrapper using poke-engine's `monte_carlo_tree_search_with_priors`.

Unlike Python-layer output re-weighting (which only reorders MCTS visits
after the search ends), this passes priors *into* the search so PUCT
selection in the tree expands more visits on plausible lines. This is the
actual lever for the search-budget-compounding question we couldn't move
with chaos / movenet output bias.

Flow per turn:
  1. Cheap 1 ms MCTS to discover the move-choice ordering for each side.
  2. Build prior arrays aligned with that ordering:
     - For moves in the net's prediction, use net prob
     - For everything else (switches, tera, no-op), use a small floor
     - Normalize each side's priors to sum to 1
  3. Call `monte_carlo_tree_search_with_priors(state, s1_priors, s2_priors,
     duration_ms)` for the full search budget.

The net's predictions are conditioned on the live state (V2 features), so
priors change per turn — not a static prior like chaos.
"""

from __future__ import annotations

import poke_engine as pe

from monotype.move_net_infer import predict_active_move_probs


def _strip_switch(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _build_aligned_priors(
    option_results,
    net_probs: dict[str, float],
    floor: float = 0.05,
) -> list[float]:
    """Given the engine's option list and net's {move_id: prob} predictions,
    return a prior array aligned with option_results, normalized to sum 1.

    Tera-variants get the floor (the engine's monotype tera filter handles
    them at move-selection time, but they're still in the option list).
    """
    priors = []
    for r in option_results:
        mc = r.move_choice
        if mc.endswith("-tera"):
            priors.append(floor * 0.1)  # explicitly discourage tera
        elif mc.startswith("switch "):
            priors.append(floor)
        elif mc == "No Move":
            priors.append(floor)
        else:
            priors.append(max(floor, net_probs.get(mc, floor)))
    s = sum(priors)
    if s > 0:
        priors = [p / s for p in priors]
    return priors


def mcts_with_net_priors(
    state,
    net_bundle: dict,
    search_ms: int,
    warmup_ms: int = 1,
):
    """Run the net-prior PUCT MCTS for one decision. Returns the raw
    MctsResult from poke-engine — caller picks moves as usual.

    `net_bundle` is the dict returned by `load_move_net()`.
    """
    s_str = state.to_string()
    # Warm-up search just to discover the option ordering.
    warm = pe.monte_carlo_tree_search(pe.State.from_string(s_str),
                                       duration_ms=warmup_ms)
    p1_probs = predict_active_move_probs(state, state.side_one, net_bundle)
    p2_probs = predict_active_move_probs(state, state.side_two, net_bundle)
    s1_priors = _build_aligned_priors(warm.side_one, p1_probs)
    s2_priors = _build_aligned_priors(warm.side_two, p2_probs)

    return pe.monte_carlo_tree_search_with_priors(
        pe.State.from_string(s_str), s1_priors, s2_priors,
        duration_ms=search_ms,
    )
