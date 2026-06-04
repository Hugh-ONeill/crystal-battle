#!/usr/bin/env python3
"""
Round-robin tournament over Gen 9 Monotype teams from a Showdown paste file.

Mirrors showdown/bench_roundrobin.py but:
  - loads teams from a `=== [gen9monotype] NAME ===`-delimited text file
  - suppresses Tera in the MCTS agent's chosen move (monotype rule: no Tera)
  - reports per-team win-rate with a Wilson 95% CI so outliers are obvious

Usage:
  .venv/bin/python showdown/bench_monotype.py \\
      --teams-file monotype/teams/teams_v3.txt --games 4 --search-ms 200
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9
from monotype.heuristics import recovery_override
from monotype.lead_picker import pick_leads, pick_leads_net, load_lead_net, reorder_team
from monotype.chaos_priors import active_move_probs, reweight_by_priors
from monotype.move_net_infer import load_move_net, predict_active_move_probs
from monotype.puct_search import mcts_with_net_priors
from monotype.endgame_solver import is_solvable_endgame, solve_endgame


_HEADER_RE = re.compile(r"^=== \[gen9monotype\] (.+?) ===\s*$", re.M)


def load_teams(path: Path) -> list[tuple[str, str]]:
    """Parse a Showdown paste file into [(team_name, team_body), ...]."""
    text = Path(path).expanduser().read_text()
    parts = _HEADER_RE.split(text)
    # parts: [preamble, name1, body1, name2, body2, ...]
    teams = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            teams.append((name, body))
    return teams


def _best_non_tera(side_results) -> str:
    """Pick the move_choice with the most visits, excluding '-tera' variants.

    Falls back to the overall-best move if every option is tera (shouldn't
    happen in practice but guards against an empty filter).
    """
    non_tera = [x for x in side_results if not x.move_choice.endswith("-tera")]
    pool = non_tera if non_tera else list(side_results)
    return max(pool, key=lambda x: x.visits).move_choice


# hazard move_choice -> (SideConditions attr, max layers). Re-setting a maxed
# hazard is a pure no-op; the flat-eval MCTS will pick it anyway (visits are
# ~tied with real moves), so we filter it out of move selection.
_HAZARD_MAX = {"stealthrock": ("stealth_rock", 1), "spikes": ("spikes", 3),
               "toxicspikes": ("toxic_spikes", 2), "stickyweb": ("sticky_web", 1)}


def _best_useful(side_results, opp_conditions) -> str:
    """Like `_best_non_tera`, but also drops hazards already maxed on the
    opponent's side (a wasted turn). Falls back if every option is filtered."""
    def is_noop(mc: str) -> bool:
        hz = _HAZARD_MAX.get(mc.replace(" ", "").lower())
        return bool(hz and getattr(opp_conditions, hz[0], 0) >= hz[1])

    non_tera = [x for x in side_results if not x.move_choice.endswith("-tera")]
    useful = [x for x in non_tera if not is_noop(x.move_choice)]
    pool = useful or non_tera or list(side_results)
    return max(pool, key=lambda x: x.visits).move_choice


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def play_one(team1_str: str, team2_str: str, search_ms: int,
             max_turns: int = 120, use_heal_heuristic: bool = False,
             heal_hp_threshold: float = 0.50,
             lead_p1: int = 0, lead_p2: int = 0,
             chaos_alpha: float = 0.0,
             move_net_path: str | None = None,
             move_net_alpha: float = 0.0,
             puct_net_path: str | None = None,
             use_endgame_solver: bool = False,
             endgame_threshold: int = 3,
             endgame_depth: int = 12) -> int:
    """Play one game; return 1 if side_one wins, 2 if side_two, 0 on draw/error.

    `lead_p1` / `lead_p2` select which mon to send out at turn 1 (0 = the
    leftmost mon in the paste, the original default).
    """
    if lead_p1:
        team1_str = reorder_team(team1_str, lead_p1)
    if lead_p2:
        team2_str = reorder_team(team2_str, lead_p2)
    state = build_pe_state_gen9(team1_str, team2_str)
    prev_str = ""
    stuck_turns = 0
    # Per-side history for the heal heuristic: hp at start of *previous* turn
    # and which mon was active then (resets the damage signal on a switch).
    prev_hp1: int | None = None
    prev_hp2: int | None = None
    prev_id1: str | None = None
    prev_id2: str | None = None
    for _ in range(max_turns):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            return 2
        if s2_alive == 0:
            return 1

        # Endgame solver: when total alive <= threshold, replace MCTS with
        # exhaustive minimax. Same move/action format as MCTS — bench loop
        # downstream doesn't need to change.
        if use_endgame_solver and is_solvable_endgame(state, max_total_alive=endgame_threshold):
            try:
                p1_solve, p2_solve, _ = solve_endgame(state, max_depth=endgame_depth)
                if p1_solve and p2_solve:
                    p1_move = p1_solve if not p1_solve.startswith("switch ") else p1_solve
                    p2_move = p2_solve if not p2_solve.startswith("switch ") else p2_solve
                    p1_clean = _strip_switch_prefix(p1_move)
                    p2_clean = _strip_switch_prefix(p2_move)
                    try:
                        instructions = pe.generate_instructions(state, p1_clean, p2_clean)
                    except Exception:
                        return 0
                    if not instructions:
                        return 0
                    roll = random.random() * 100
                    cum = 0.0
                    chosen = instructions[0]
                    for inst in instructions:
                        cum += inst.percentage
                        if roll <= cum:
                            chosen = inst
                            break
                    a1 = state.side_one.pokemon[int(state.side_one.active_index)]
                    a2 = state.side_two.pokemon[int(state.side_two.active_index)]
                    prev_hp1, prev_id1 = a1.hp, a1.id
                    prev_hp2, prev_id2 = a2.hp, a2.id
                    state = state.apply_instructions(chosen)
                    cur_str = state.to_string()
                    if cur_str == prev_str:
                        stuck_turns += 1
                        if stuck_turns >= 3:
                            return 0
                    else:
                        stuck_turns = 0
                    prev_str = cur_str
                    continue
            except Exception:
                pass  # fall through to MCTS if solver errors

        s_str = state.to_string()
        try:
            if puct_net_path:
                # Both sides use the same PUCT-with-net-priors search. The
                # underlying tree models both players, so one shared search
                # per side is equivalent to the two-independent-trees pattern
                # below, but with net-biased PUCT throughout.
                bundle = load_move_net(puct_net_path)
                r1 = mcts_with_net_priors(state, bundle, search_ms=search_ms)
                r2 = mcts_with_net_priors(state, bundle, search_ms=search_ms)
            else:
                ps1 = pe.State.from_string(s_str)
                r1 = pe.monte_carlo_tree_search(ps1, duration_ms=search_ms)
                ps2 = pe.State.from_string(s_str)
                r2 = pe.monte_carlo_tree_search(ps2, duration_ms=search_ms)
        except Exception:
            return 0

        # Build per-side priors as merged dict {move_id: prob}, drawing from
        # whichever sources are enabled. Then re-weight MCTS visits by these.
        # Multiple priors stack additively on the visit count formula.
        p1_priors: dict[str, float] = {}
        p2_priors: dict[str, float] = {}
        if chaos_alpha > 0.0:
            for k, v in active_move_probs(state.side_one).items():
                p1_priors[k] = p1_priors.get(k, 0.0) + chaos_alpha * v
            for k, v in active_move_probs(state.side_two).items():
                p2_priors[k] = p2_priors.get(k, 0.0) + chaos_alpha * v
        if move_net_path and move_net_alpha > 0.0:
            mb = load_move_net(move_net_path)
            for k, v in predict_active_move_probs(state, state.side_one, mb).items():
                p1_priors[k] = p1_priors.get(k, 0.0) + move_net_alpha * v
            for k, v in predict_active_move_probs(state, state.side_two, mb).items():
                p2_priors[k] = p2_priors.get(k, 0.0) + move_net_alpha * v

        if p1_priors or p2_priors:
            # alpha=1.0 here because the per-source alphas already scaled
            # the prior values when building p1_priors / p2_priors above.
            scored1 = reweight_by_priors(r1.side_one, p1_priors, alpha=1.0)
            scored2 = reweight_by_priors(r2.side_two, p2_priors, alpha=1.0)
            p1_move = max(scored1, key=scored1.get) if scored1 \
                else _best_useful(r1.side_one, state.side_two.side_conditions)
            p2_move = max(scored2, key=scored2.get) if scored2 \
                else _best_useful(r2.side_two, state.side_one.side_conditions)
        else:
            p1_move = _best_useful(r1.side_one, state.side_two.side_conditions)
            p2_move = _best_useful(r2.side_two, state.side_one.side_conditions)

        # Heuristic override: prefer recovery if the active is low HP and
        # the last turn's chip didn't out-damage the heal.
        if use_heal_heuristic:
            ov1 = recovery_override(state.side_one, r1.side_one, prev_hp1, prev_id1,
                                    hp_threshold=heal_hp_threshold)
            if ov1 is not None:
                p1_move = ov1
            ov2 = recovery_override(state.side_two, r2.side_two, prev_hp2, prev_id2,
                                    hp_threshold=heal_hp_threshold)
            if ov2 is not None:
                p2_move = ov2

        # Snapshot current active id/hp for next turn's damage delta.
        a1 = state.side_one.pokemon[int(state.side_one.active_index)]
        a2 = state.side_two.pokemon[int(state.side_two.active_index)]
        prev_hp1, prev_id1 = a1.hp, a1.id
        prev_hp2, prev_id2 = a2.hp, a2.id

        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "No Move" and p2_move == "No Move":
            return 0
        p1_clean = _strip_switch_prefix(p1_move)
        p2_clean = _strip_switch_prefix(p2_move)

        try:
            instructions = pe.generate_instructions(state, p1_clean, p2_clean)
        except Exception:
            return 0
        if not instructions:
            return 0

        roll = random.random() * 100
        cum = 0.0
        chosen = instructions[0]
        for inst in instructions:
            cum += inst.percentage
            if roll <= cum:
                chosen = inst
                break

        state = state.apply_instructions(chosen)
        cur_str = state.to_string()
        if cur_str == prev_str:
            stuck_turns += 1
            if stuck_turns >= 3:
                return 0
        else:
            stuck_turns = 0
        prev_str = cur_str
    return 0


# multiprocessing payload: (a_idx, b_idx, p1_idx, p2_idx, search_ms, seed,
#                           heal_p1, heal_p2)
_TEAMS: list[tuple[str, str]] = []  # set per-process in _init_worker


def _init_worker(teams):
    global _TEAMS
    _TEAMS = teams


def _worker(task):
    (a, b, p1, p2, search_ms, seed, heal_p1, heal_p2, heal_hp_thresh,
     lead_p1, lead_p2, chaos_alpha, move_net_path, move_net_alpha,
     puct_net_path, use_endgame, endgame_th, endgame_d) = task
    if seed is not None:
        random.seed(seed)
    use_heal = heal_p1 or heal_p2
    r = play_one(_TEAMS[p1][1], _TEAMS[p2][1], search_ms,
                 use_heal_heuristic=use_heal,
                 heal_hp_threshold=heal_hp_thresh,
                 lead_p1=lead_p1, lead_p2=lead_p2,
                 chaos_alpha=chaos_alpha,
                 move_net_path=move_net_path,
                 move_net_alpha=move_net_alpha,
                 puct_net_path=puct_net_path,
                 use_endgame_solver=use_endgame,
                 endgame_threshold=endgame_th,
                 endgame_depth=endgame_d)
    return (a, b, p1, p2, r)


def wilson_ci(wins: int, decided: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if decided == 0:
        return (0.0, 0.0)
    n = decided
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser = argparse.ArgumentParser(description="Monotype round-robin bench")
    default_teams = Path(__file__).parent.parent / "monotype" / "teams" / "teams_v3.txt"
    parser.add_argument("--teams-file", type=str, default=str(default_teams),
                        help="path to Showdown paste file with monotype teams")
    parser.add_argument("--games", type=int, default=4,
                        help="games per direction per pair (total per pair = 2*games)")
    parser.add_argument("--search-ms", type=int, default=200)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-mirrors", action="store_true")
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--use-heal-heuristic", action="store_true",
                        help="both sides: override MCTS with a 50%%-heal move "
                             "when active HP<threshold and last hit < heal amount")
    parser.add_argument("--heal-hp-threshold", type=float, default=0.50,
                        help="HP fraction below which heal heuristic may fire "
                             "(default 0.50)")
    parser.add_argument("--use-lead-picker", action="store_true",
                        help="pick leads via 6x6 MCTS-eval maximin instead of "
                             "using paste-order leftmost")
    parser.add_argument("--lead-pick-ms", type=int, default=100,
                        help="MCTS budget per cell when picking leads (default 100)")
    parser.add_argument("--lead-net", type=str, default=None,
                        help="path to trained LeadPickerNet .pt; if set, "
                             "supersedes --use-lead-picker and picks via net "
                             "inference (microseconds, no MCTS)")
    parser.add_argument("--chaos-alpha", type=float, default=0.0,
                        help="if >0, re-weight MCTS visits at each side's "
                             "active by Smogon chaos prior * alpha. "
                             "0=pure MCTS (default); 1=equal weight to a "
                             "100%% prior and the full visit count")
    parser.add_argument("--move-net", type=str, default=None,
                        help="path to trained MoveNet .pt for state-conditioned "
                             "move-prediction bias")
    parser.add_argument("--move-net-alpha", type=float, default=0.0,
                        help="if >0 (and --move-net set), re-weight MCTS visits "
                             "at each side's active by net prediction * alpha. "
                             "Stacks additively with --chaos-alpha")
    parser.add_argument("--puct-net", type=str, default=None,
                        help="path to MoveNet .pt to use as PUCT priors INSIDE "
                             "the MCTS search (poke-engine's "
                             "monte_carlo_tree_search_with_priors). Real lever — "
                             "biases the search itself, not just output reorder")
    parser.add_argument("--use-endgame-solver", action="store_true",
                        help="when total alive <= threshold, replace MCTS with "
                             "exhaustive simultaneous-move minimax solver")
    parser.add_argument("--endgame-threshold", type=int, default=3,
                        help="trigger solver when total alive <= this (default 3: "
                             "covers 1v1 / 2v1 / 1v2; 4 adds 2v2 but is slower)")
    parser.add_argument("--endgame-depth", type=int, default=12,
                        help="max recursion depth for the endgame solver")
    args = parser.parse_args()

    teams = load_teams(Path(args.teams_file))
    if len(teams) < 2:
        print(f"only {len(teams)} teams parsed from {args.teams_file}", file=sys.stderr)
        sys.exit(1)

    n = len(teams)
    team_ids = list(range(n))

    pairs: list[tuple[int, int]] = []
    for a, b in combinations_with_replacement(team_ids, 2):
        if a == b and not args.include_mirrors:
            continue
        pairs.append((a, b))

    # Pre-compute lead picks per ordered (P1_team, P2_team) pair so the
    # picker (MCTS or net) only runs once per matchup instead of per game.
    lead_cache: dict[tuple[int, int], tuple[int, int]] = {}
    use_picker = args.use_lead_picker or args.lead_net
    if use_picker:
        ordered_pairs = set()
        for a, b in pairs:
            ordered_pairs.add((a, b))
            if a != b:
                ordered_pairs.add((b, a))
        net = None
        if args.lead_net:
            net = load_lead_net(args.lead_net)
            print(f"=== picking leads for {len(ordered_pairs)} ordered pairs "
                  f"via net ({args.lead_net}) ===")
        else:
            print(f"=== picking leads for {len(ordered_pairs)} ordered pairs "
                  f"via MCTS @ {args.lead_pick_ms}ms/cell "
                  f"({len(ordered_pairs)*36*args.lead_pick_ms/1000:.0f}s expected) ===")
        t_lead = time.time()
        for k, (p1, p2) in enumerate(sorted(ordered_pairs)):
            if net is not None:
                lp1, lp2, _ = pick_leads_net(teams[p1][1], teams[p2][1], net)
            else:
                lp1, lp2, _ = pick_leads(teams[p1][1], teams[p2][1],
                                         search_ms=args.lead_pick_ms)
            lead_cache[(p1, p2)] = (lp1, lp2)
            if (k + 1) % max(1, len(ordered_pairs) // 10) == 0:
                print(f"  [{k+1}/{len(ordered_pairs)}] {time.time()-t_lead:.0f}s",
                      flush=True)

    tasks = []
    heal = bool(args.use_heal_heuristic)
    heal_th = float(args.heal_hp_threshold)
    chaos_a = float(args.chaos_alpha)
    mn_path = args.move_net
    mn_alpha = float(args.move_net_alpha)
    puct_path = args.puct_net
    use_eg = bool(args.use_endgame_solver)
    eg_th = int(args.endgame_threshold)
    eg_d = int(args.endgame_depth)
    for (a, b) in pairs:
        for i in range(args.games):
            seed = (args.seed + i) if args.seed is not None else None
            # counterbalance: half with a as P1, half with b as P1
            la_ab, lb_ab = lead_cache.get((a, b), (0, 0))
            lb_ba, la_ba = lead_cache.get((b, a), (0, 0))
            tasks.append((a, b, a, b, args.search_ms, seed, heal, heal, heal_th,
                          la_ab, lb_ab, chaos_a, mn_path, mn_alpha, puct_path,
                          use_eg, eg_th, eg_d))
            tasks.append((a, b, b, a, args.search_ms, seed, heal, heal, heal_th,
                          lb_ba, la_ba, chaos_a, mn_path, mn_alpha, puct_path,
                          use_eg, eg_th, eg_d))

    print(f"=== monotype round-robin: {n} teams, {len(pairs)} pairs, "
          f"{2*args.games} games/pair, {args.search_ms}ms search, "
          f"{args.workers} workers, {len(tasks)} total games ===")
    for i, (name, _) in enumerate(teams):
        print(f"  [{i:2d}] {name}")

    pair_stats: dict[tuple[int, int], dict[str, int]] = {
        pair: {"a_wins": 0, "b_wins": 0, "draws": 0} for pair in pairs
    }

    t0 = time.time()
    completed = 0
    if args.workers <= 1:
        _init_worker(teams)
        results = (_worker(t) for t in tasks)
        pool = None
    else:
        pool = mp.Pool(processes=args.workers,
                       initializer=_init_worker, initargs=(teams,))
        results = pool.imap_unordered(_worker, tasks)

    progress_step = max(1, len(tasks) // 20)
    for (a, b, p1, p2, r) in results:
        completed += 1
        s = pair_stats[(a, b)]
        if r == 0:
            s["draws"] += 1
        else:
            winner_team = p1 if r == 1 else p2
            if winner_team == a:
                s["a_wins"] += 1
            else:
                s["b_wins"] += 1
        if completed % progress_step == 0:
            print(f"  [{completed}/{len(tasks)}] {time.time()-t0:.0f}s",
                  flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    def short(name: str, w: int = 22) -> str:
        return (name[:w-1] + "…") if len(name) > w else name

    # aggregate per-team
    team_wins = [0] * n
    team_losses = [0] * n
    team_draws = [0] * n
    print("\n=== Per-pair results ===")
    for (a, b), s in sorted(pair_stats.items()):
        decided = s["a_wins"] + s["b_wins"]
        a_pct = (s["a_wins"] / decided * 100) if decided else 0
        total = decided + s["draws"]
        print(f"  {short(teams[a][0]):>22} vs {short(teams[b][0]):<22}: "
              f"{s['a_wins']:3d}W {s['b_wins']:3d}L {s['draws']:3d}D "
              f"({a_pct:5.1f}% for {short(teams[a][0])}, {decided}/{total} decided)")
        if a == b:
            continue
        team_wins[a] += s["a_wins"]
        team_losses[a] += s["b_wins"]
        team_draws[a] += s["draws"]
        team_wins[b] += s["b_wins"]
        team_losses[b] += s["a_wins"]
        team_draws[b] += s["draws"]

    rows = []
    for i in range(n):
        w, l, d = team_wins[i], team_losses[i], team_draws[i]
        decided = w + l
        pct = (w / decided * 100) if decided else 0.0
        lo, hi = wilson_ci(w, decided)
        rows.append((pct, w, l, d, lo, hi, i))
    rows.sort(reverse=True)

    print("\n=== Per-team standings (Wilson 95% CI on decided games) ===")
    print(f"  {'team':>22}  {'win%':>6}  {'95% CI':>15}  "
          f"{'W':>4}/{'L':>4}/{'D':>4}  decided%")
    for pct, w, l, d, lo, hi, i in rows:
        total = w + l + d
        decided_frac = ((w + l) / total * 100) if total else 0
        ci_str = f"[{lo*100:4.1f},{hi*100:5.1f}]"
        print(f"  {short(teams[i][0]):>22}  {pct:5.1f}%  {ci_str:>15}  "
              f"{w:4d}/{l:4d}/{d:4d}  {decided_frac:5.1f}%")

    print("\nNote: Tera-suppressed (monotype rule). Move choices ending in "
          "'-tera' were filtered before each turn.")


if __name__ == "__main__":
    main()
