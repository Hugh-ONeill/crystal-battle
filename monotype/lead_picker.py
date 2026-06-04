"""
Team-preview lead picker — bridges the gap where bench_monotype currently
sends in the leftmost mon from the paste as turn-1 active.

In real Showdown, both players see each other's 6-mon roster at preview and
pick one to lead. The pick is a 6×6 simultaneous-move sub-game: each side
picks a lead independently, payoff is `pe.evaluate(state)` from the
resulting turn-1 state.

We solve this with a static-eval matrix + pure maximin (P1 picks the lead
whose worst-case opp counter is best; P2 picks the lead whose worst-case
own counter from P1 is least-favorable). This is a coarse first cut:

  - skips mixed-strategy equilibrium when the matrix has no saddle point
  - uses static eval instead of brief MCTS (cheap, ~36 ms total per game)
  - doesn't account for "leading X to scout Y's set" — only direct matchup

Upgrade paths: replace `pe.evaluate` with a 50-100 ms root MCTS per cell;
fall back to mixed strategy via linear programming when maximin != minimax;
learn a small lead-policy net from self-play.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9


def split_team_body(body: str) -> list[str]:
    """Split a 6-mon Showdown paste body into 6 individual mon blocks.

    Mon blocks in a paste are separated by blank lines; the canonical
    format is "Species @ Item\\nAbility: ...\\n...\\n- Move1\\n...".
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body.strip()) if b.strip()]
    return blocks


def reorder_team(body: str, lead_idx: int) -> str:
    """Return body with the block at `lead_idx` moved to position 0."""
    blocks = split_team_body(body)
    if lead_idx == 0 or lead_idx >= len(blocks):
        return body
    return "\n\n".join([blocks[lead_idx]] + blocks[:lead_idx] + blocks[lead_idx + 1:])


def species_of(block: str) -> str:
    """Best-effort: return the species name from the first line of a paste block."""
    first = block.splitlines()[0].split("@")[0].strip()
    # strip "(Gender)" or "(form)" tail if present
    first = re.sub(r"\s*\([^)]*\)\s*$", "", first).strip()
    return first


def _root_value_from_mcts(result) -> float:
    """Visit-weighted average score at MCTS root, from side_one's perspective.

    Each move-entry has (total_score, visits); the sum-of-scores / sum-of-visits
    over all root moves is the MCTS estimate of the position value.
    """
    total_score = 0.0
    total_visits = 0
    for m in result.side_one:
        if m.visits > 0:
            total_score += m.total_score
            total_visits += m.visits
    if total_visits == 0:
        return 0.0
    return total_score / total_visits


def build_eval_matrix(
    team1_body: str,
    team2_body: str,
    *,
    search_ms: int = 100,
    use_static_eval: bool = False,
) -> list[list[float]]:
    """6x6 matrix of position values for each (P1 lead, P2 lead) pairing.

    Positive entries favor P1. When `use_static_eval` is True, falls back
    to `pe.evaluate(state)` — fast but doesn't differentiate at turn 1
    since both teams are full-HP equivalent. Default uses a brief root MCTS.
    """
    blocks1 = split_team_body(team1_body)
    blocks2 = split_team_body(team2_body)
    n1, n2 = len(blocks1), len(blocks2)
    matrix = [[0.0] * n2 for _ in range(n1)]
    for i in range(n1):
        for j in range(n2):
            t1 = reorder_team(team1_body, i)
            t2 = reorder_team(team2_body, j)
            try:
                state = build_pe_state_gen9(t1, t2)
                if use_static_eval:
                    matrix[i][j] = float(pe.evaluate(state))
                else:
                    r = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
                    matrix[i][j] = _root_value_from_mcts(r)
            except Exception:
                matrix[i][j] = 0.0
    return matrix


def maximin_lead_pair(matrix: list[list[float]]) -> tuple[int, int]:
    """Pure-strategy maximin for P1, minimax for P2.

    P1 maximises its worst-case eval (over P2's choice of column).
    P2 minimises its worst-case eval (over P1's choice of row), since the
    matrix is from P1's perspective and P2 wants P1's eval low.

    Returns (p1_lead_idx, p2_lead_idx). When the matrix has no saddle point,
    these two pure choices are still a reasonable first cut.
    """
    n1 = len(matrix)
    n2 = len(matrix[0]) if matrix else 0
    if n1 == 0 or n2 == 0:
        return (0, 0)

    row_mins = [min(matrix[i]) for i in range(n1)]
    p1_lead = max(range(n1), key=lambda i: row_mins[i])

    col_maxes = [max(matrix[i][j] for i in range(n1)) for j in range(n2)]
    p2_lead = min(range(n2), key=lambda j: col_maxes[j])

    return p1_lead, p2_lead


def pick_leads(
    team1_body: str,
    team2_body: str,
    *,
    search_ms: int = 100,
    use_static_eval: bool = False,
) -> tuple[int, int, list[list[float]]]:
    """Convenience: returns (p1_lead_idx, p2_lead_idx, eval_matrix)."""
    matrix = build_eval_matrix(team1_body, team2_body,
                               search_ms=search_ms, use_static_eval=use_static_eval)
    p1, p2 = maximin_lead_pair(matrix)
    return p1, p2, matrix


# ---------------------------------------------------------------------
# Net-based picker — trained on Smogon replay leads (see lead_net.py +
# train_lead_net.py). Inference is microseconds; caller passes the loaded
# net and a CPU/CUDA device. See `pick_leads_net()`.
# ---------------------------------------------------------------------

def pick_leads_net(team1_body: str, team2_body: str, net, device: str = "cpu"):
    """Use a trained LeadPickerNet to pick (p1_lead, p2_lead).

    Two forward passes: net(p1, p2) → p1's lead; net(p2, p1) → p2's lead.
    Returns (p1_lead_idx, p2_lead_idx, scores_matrix) where scores_matrix
    is (2, 6) — the per-mon softmax for each side (P1 first row, P2 second).
    """
    import torch
    from monotype.featurizer_lead_preview import featurize_preview

    x1, x2 = featurize_preview(team1_body, team2_body)
    tx1 = torch.from_numpy(x1).unsqueeze(0).to(device)  # (1, 6, MON_DIM)
    tx2 = torch.from_numpy(x2).unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        logits_p1 = net(tx1, tx2).squeeze(0)  # (6,)
        logits_p2 = net(tx2, tx1).squeeze(0)
        probs_p1 = torch.softmax(logits_p1, dim=-1)
        probs_p2 = torch.softmax(logits_p2, dim=-1)
    p1_lead = int(probs_p1.argmax().item())
    p2_lead = int(probs_p2.argmax().item())
    scores = [probs_p1.cpu().tolist(), probs_p2.cpu().tolist()]
    return p1_lead, p2_lead, scores


def pick_leads_hybrid(
    team1_body: str,
    team2_body: str,
    net,
    *,
    top_k: int = 2,
    search_ms: int = 100,
    device: str = "cpu",
):
    """Net-pruned MCTS lead picker: net proposes, search disposes.

    The imitation net is good at *narrowing* the lead to a few plausible mons
    but unreliable at the final pick (it imitates humans, who don't always lead
    engine-optimally — e.g. it leads Archaludon for Steel where Heatran wins).
    So: take each side's top-`top_k` net candidates, run a brief root MCTS over
    just that `k x k` sub-grid, and resolve it with maximin. Costs `k*k` cells
    instead of the full 36, while letting search correct the net's bad pick.

    Returns (p1_lead_idx, p2_lead_idx, info) where info carries the candidate
    indices, their net probabilities, and the evaluated sub-matrix — mapped
    back to original (pre-prune) team indices.
    """
    _, _, scores = pick_leads_net(team1_body, team2_body, net, device=device)
    probs_p1, probs_p2 = scores[0], scores[1]

    n1 = len(split_team_body(team1_body))
    n2 = len(split_team_body(team2_body))
    cand1 = sorted(range(n1), key=lambda i: probs_p1[i], reverse=True)[:min(top_k, n1)]
    cand2 = sorted(range(n2), key=lambda j: probs_p2[j], reverse=True)[:min(top_k, n2)]

    sub = [[0.0] * len(cand2) for _ in range(len(cand1))]
    for a, i in enumerate(cand1):
        for b, j in enumerate(cand2):
            try:
                state = build_pe_state_gen9(reorder_team(team1_body, i),
                                            reorder_team(team2_body, j))
                r = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
                sub[a][b] = _root_value_from_mcts(r)
            except Exception:
                sub[a][b] = 0.0

    ra, rb = maximin_lead_pair(sub)
    p1_lead, p2_lead = cand1[ra], cand2[rb]
    info = {
        "cand1": cand1, "cand2": cand2,
        "net_probs_p1": probs_p1, "net_probs_p2": probs_p2,
        "submatrix": sub,
    }
    return p1_lead, p2_lead, info


def load_lead_net(weights_path: str, device: str = "cpu"):
    """Load a trained LeadPickerNet from a .pt checkpoint."""
    import torch
    from monotype.lead_net import LeadPickerNet
    net = LeadPickerNet()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def fmt_matrix(matrix: list[list[float]], team1_body: str, team2_body: str) -> str:
    """Pretty-print the 6x6 lead-eval matrix with species labels."""
    b1 = split_team_body(team1_body)
    b2 = split_team_body(team2_body)
    names1 = [species_of(b) for b in b1]
    names2 = [species_of(b) for b in b2]
    out = []
    header = "             " + " ".join(f"{n[:10]:>10}" for n in names2)
    out.append(header)
    for i, row in enumerate(matrix):
        cells = " ".join(f"{v:+10.1f}" for v in row)
        out.append(f"  {names1[i][:11]:<11} {cells}")
    return "\n".join(out)
