# AlphaZero-style training loop: search -> train -> search -> train
# Uses lookahead search to generate expert play, trains policy to match
#
# With --rust flag, uses crystal_engine_rs for 280-511x faster search,
# enabling practical 2-ply search during training.
#
# Usage:
#   .venv/bin/python training/search_train.py --rust --depth 2 --iterations 10

from __future__ import annotations

import argparse
import copy
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.types import TypeChart
from engine.data_loader import DataStore
from engine.battle_state import BattleState
from engine.player_state import PlayerState
from engine.turn_engine import resolve_turn, resolve_forced_switches
from engine.actions import Switch, UseMove, Struggle
from training.baselines import SmartAgent, MaxDamageAgent
from training.maskable_recurrent_ppo import MaskableRecurrentPPO
from training.opponent_model import OpponentPredictor
from training.lookahead import LookaheadAgent, _int_to_action
from training.evaluate import evaluate_vs_baseline
from gym_env.team_builder import build_team
from gym_env.obs_builder import build_observation, OBS_SIZE


def play_search_games(
    agent: LookaheadAgent,
    data: DataStore,
    tc: TypeChart,
    n_games: int = 500,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Play games with lookahead and record sequences."""
    sequences = []
    rng = random.Random(seed)

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**31)

        # opponent is Smart or MaxDmg
        if random.Random(game_seed).random() < 0.5:
            opp = SmartAgent(tc, seed=game_seed + 1000)
        else:
            opp = MaxDamageAgent(tc)

        t1 = build_team(data, rng=random.Random(game_seed + 100), tier="ou")
        t2 = build_team(data, rng=random.Random(game_seed + 200), tier="ou")
        battle = BattleState(
            p1=PlayerState(team=t1), p2=PlayerState(team=t2),
            rng=random.Random(game_seed + 300),
        )

        agent.reset()
        obs_seq = []
        action_seq = []
        mask_seq = []

        for turn in range(100):
            if battle.is_over:
                break

            obs = build_observation(
                battle.p1, battle.p2, tc, turn=battle.turn,
                weather=battle.weather, weather_turns=battle.weather_turns,
            )
            mask = np.array(
                battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                dtype=np.float32,
            )

            # lookahead picks the action
            action = agent.act(battle)

            # convert to int
            if isinstance(action, UseMove):
                action_int = action.slot_index
            elif isinstance(action, Switch):
                action_int = 4 + action.team_index
            else:
                action_int = 0

            obs_seq.append(obs)
            action_seq.append(action_int)
            mask_seq.append(mask)

            # opponent acts
            a2 = opp.act(battle.p2, battle.p1)
            resolve_turn(battle, action, a2, tc)

            # forced switches
            sw1 = sw2 = None
            if battle.p1.must_switch:
                for i, p in enumerate(battle.p1.team):
                    if i != battle.p1.active_index and not p.is_fainted:
                        sw1 = Switch(team_index=i)
                        break
                if sw1:
                    obs_sw = build_observation(
                        battle.p1, battle.p2, tc, turn=battle.turn,
                        weather=battle.weather, weather_turns=battle.weather_turns,
                    )
                    mask_sw = np.array(
                        battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                        dtype=np.float32,
                    )
                    obs_seq.append(obs_sw)
                    action_seq.append(4 + sw1.team_index)
                    mask_seq.append(mask_sw)

            if battle.p2.must_switch:
                sw2_a = opp.act(battle.p2, battle.p1)
                sw2 = sw2_a if isinstance(sw2_a, Switch) else None
                if sw2 is None:
                    for i, p in enumerate(battle.p2.team):
                        if i != battle.p2.active_index and not p.is_fainted:
                            sw2 = Switch(team_index=i); break
            if sw1 or sw2:
                resolve_forced_switches(battle, sw1, sw2)

        if len(obs_seq) >= 3:
            sequences.append((
                np.array(obs_seq, dtype=np.float32),
                np.array(action_seq, dtype=np.int64),
                np.array(mask_seq, dtype=np.float32),
            ))

        if (game_idx + 1) % 100 == 0:
            print(f"    {game_idx + 1}/{n_games} games")

    return sequences


def train_on_sequences(
    model_path: str,
    sequences: list,
    epochs: int = 10,
    lr: float = 5e-4,
    device: str = "cpu",
    value_coef: float = 0.5,
):
    """Train the full model (extractor + LSTM + policy head + value head).

    Dual loss: policy_loss (cross-entropy on search actions) +
    value_loss (MSE on game outcome). Every position in a won game
    gets target +1, lost game -1, draw 0.

    Sequences can be 3-tuples (obs, action, mask) for backward compat
    or 4-tuples (obs, action, mask, outcome) for value training.
    """
    from training.attention_extractor import AttentionFeatureExtractor
    import gymnasium

    # load existing model weights
    checkpoint = torch.load(f"{model_path}_weights.pt", map_location=device, weights_only=True)
    features_dim = checkpoint["features_dim"]
    lstm_hidden = checkpoint["lstm_hidden"]
    net_arch = checkpoint["net_arch"]

    obs_space = gymnasium.spaces.Box(low=-10, high=10, shape=(OBS_SIZE,), dtype=np.float32)
    extractor = AttentionFeatureExtractor(obs_space, features_dim=features_dim)
    lstm = nn.LSTM(features_dim, lstm_hidden, num_layers=1, batch_first=True)

    # policy head
    head_layers = []
    in_dim = lstm_hidden
    for h in net_arch:
        head_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    head_layers.append(nn.Linear(in_dim, 10))
    policy_head = nn.Sequential(*head_layers)

    # value head: lstm_hidden -> 256 -> 1, tanh output
    value_head = nn.Sequential(
        nn.Linear(lstm_hidden, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Tanh(),
    )

    # load weights
    extractor.load_state_dict(checkpoint["extractor"])
    lstm.load_state_dict(checkpoint["lstm"])
    policy_head.load_state_dict(checkpoint["policy_head"])
    if "value_head" in checkpoint:
        value_head.load_state_dict(checkpoint["value_head"])
    else:
        print("    (no value_head in checkpoint, initializing fresh)")

    extractor = extractor.to(device)
    lstm = lstm.to(device)
    policy_head = policy_head.to(device)
    value_head = value_head.to(device)

    # freeze extractor and LSTM -- only train heads
    # the imitation-pretrained features are good; search training
    # should only adapt the decision layers, not distort representations
    for param in extractor.parameters():
        param.requires_grad = False
    for param in lstm.parameters():
        param.requires_grad = False

    head_params = list(policy_head.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.Adam(head_params, lr=lr)

    # normalize sequences to 4-tuples
    normalized = []
    for seq in sequences:
        if len(seq) == 4:
            normalized.append(seq)
        else:
            # legacy 3-tuple: no outcome, use 0.0 (won't contribute to value loss)
            normalized.append((seq[0], seq[1], seq[2], 0.0))

    has_outcomes = sum(1 for s in normalized if s[3] != 0.0)
    print(f"    {len(normalized)} sequences, {has_outcomes} with outcomes")

    # split
    rng = random.Random(42)
    rng.shuffle(normalized)
    val_size = max(1, len(normalized) // 10)
    val_seqs = normalized[:val_size]
    train_seqs = normalized[val_size:]

    for epoch in range(epochs):
        extractor.train(); lstm.train(); policy_head.train(); value_head.train()
        train_ploss = 0; train_vloss = 0
        train_correct = 0; train_count = 0

        indices = list(range(len(train_seqs)))
        random.shuffle(indices)

        batch_size = 16
        optimizer.zero_grad()

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_loss = 0

            for idx in batch_idx:
                obs_seq, act_seq, mask_seq, outcome = train_seqs[idx]
                obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
                act_t = torch.tensor(act_seq, dtype=torch.long, device=device)
                mask_t = torch.tensor(mask_seq, dtype=torch.float32, device=device)

                # shared forward pass
                features = extractor(obs_t).unsqueeze(0)  # (1, T, 256)
                lstm_out, _ = lstm(features)               # (1, T, 256)
                hidden = lstm_out.squeeze(0)               # (T, 256)

                # policy loss
                logits = policy_head(hidden)
                logits = logits + (1 - mask_t) * -1e9
                policy_loss = F.cross_entropy(logits, act_t)

                # value loss (only if we have an outcome)
                if outcome != 0.0:
                    value_pred = value_head(hidden).squeeze(-1)  # (T,)
                    value_target = torch.full_like(value_pred, outcome)
                    v_loss = F.mse_loss(value_pred, value_target)
                else:
                    v_loss = torch.tensor(0.0, device=device)

                loss = policy_loss + value_coef * v_loss
                batch_loss = batch_loss + loss / len(batch_idx)

                preds = logits.argmax(dim=1)
                train_correct += (preds == act_t).sum().item()
                train_count += len(act_t)
                train_ploss += policy_loss.item() * len(act_t)
                train_vloss += v_loss.item() * len(act_t)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # validate
        extractor.eval(); lstm.eval(); policy_head.eval(); value_head.eval()
        val_correct = 0; val_count = 0; val_vloss = 0
        with torch.no_grad():
            for obs_seq, act_seq, mask_seq, outcome in val_seqs:
                obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
                act_t = torch.tensor(act_seq, dtype=torch.long, device=device)
                mask_t = torch.tensor(mask_seq, dtype=torch.float32, device=device)

                features = extractor(obs_t).unsqueeze(0)
                lstm_out, _ = lstm(features)
                hidden = lstm_out.squeeze(0)

                logits = policy_head(hidden)
                logits = logits + (1 - mask_t) * -1e9
                preds = logits.argmax(dim=1)
                val_correct += (preds == act_t).sum().item()
                val_count += len(act_t)

                if outcome != 0.0:
                    vp = value_head(hidden).squeeze(-1)
                    vt = torch.full_like(vp, outcome)
                    val_vloss += F.mse_loss(vp, vt).item() * len(act_t)

        avg_ploss = train_ploss / max(train_count, 1)
        avg_vloss = train_vloss / max(train_count, 1)
        print(f"    Epoch {epoch+1}: acc={train_correct/train_count:.3f} "
              f"ploss={avg_ploss:.4f} vloss={avg_vloss:.4f} "
              f"val_acc={val_correct/val_count:.3f}")

    # save updated weights (including value head)
    torch.save({
        "extractor": extractor.state_dict(),
        "lstm": lstm.state_dict(),
        "policy_head": policy_head.state_dict(),
        "value_head": value_head.state_dict(),
        "features_dim": features_dim,
        "lstm_hidden": lstm_hidden,
        "net_arch": net_arch,
    }, f"{model_path}_weights.pt")


def save_model_weights(policy_path: str, out_path: str, device: str = "cpu"):
    """Extract weights from a PPO model into the standalone format."""
    model = MaskableRecurrentPPO.load(policy_path, device=device)

    extractor_state = model.policy.features_extractor.state_dict()
    lstm_state = model.policy.lstm_actor.state_dict()

    # reconstruct policy head from mlp_extractor + action_net
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    # build head state dict
    head_state = {}
    for key, param in policy_net.state_dict().items():
        head_state[key] = param
    # add action_net as the final layer
    n_hidden = len([k for k in policy_net.state_dict() if 'weight' in k])
    final_idx = n_hidden * 2  # each hidden has weight + bias, ReLU has none
    head_state[f"{final_idx}.weight"] = action_net.weight.data
    head_state[f"{final_idx}.bias"] = action_net.bias.data

    # extract value head from PPO's critic path
    value_net = model.policy.mlp_extractor.value_net
    value_output = model.policy.value_net
    value_head_state = {
        "0.weight": value_net[0].weight.data,
        "0.bias": value_net[0].bias.data,
        # value_net is Sequential(Linear, Tanh, Linear, Tanh)
        # we want: Linear(256, 256) + ReLU + Linear(256, 1) + Tanh
        # map PPO's value_net[0] -> our value_head[0], value_output -> our value_head[2]
        "2.weight": value_output.weight.data,
        "2.bias": value_output.bias.data,
    }

    torch.save({
        "extractor": extractor_state,
        "lstm": lstm_state,
        "policy_head": head_state,
        "value_head": value_head_state,
        "features_dim": 256,
        "lstm_hidden": 256,
        "net_arch": [256, 256],
    }, out_path)
    print(f"Saved model weights to {out_path}")


def evaluate_current(policy_path: str, opp_model_path: str,
                     tc: TypeChart, device: str = "cpu"):
    """Quick evaluation of current model with and without lookahead."""
    model = MaskableRecurrentPPO.load(policy_path, device=device)

    # without lookahead
    print("  Without lookahead:")
    for bl in ("max_damage", "smart"):
        r = evaluate_vs_baseline(model, bl, n_games=100, seed=42, both_sides=False)
        print(f"    vs {bl:12s}: {r['win_rate']:.1%}")

    # with lookahead
    opp_model = OpponentPredictor()
    opp_model.load_state_dict(torch.load(opp_model_path, map_location=device, weights_only=True))
    opp_model.eval()

    from training.lookahead import evaluate_lookahead
    print("  With 1-ply lookahead:")
    evaluate_lookahead(policy_path, opp_model_path, n_games=100, depth=1, device=device)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Search-based training loop")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=500)
    parser.add_argument("--epochs-per-iter", type=int, default=10)
    parser.add_argument("--policy", type=str, default="imitation_ppo")
    parser.add_argument("--opp-model", type=str, default="opp_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rust", action="store_true", help="Use Rust engine for search (280-511x faster)")
    parser.add_argument("--depth", type=int, default=1, help="Search depth (2 practical with --rust)")
    args = parser.parse_args()

    # seed the buffer with original imitation data to prevent forgetting
    all_sequences = []
    imitation_data = Path("expert_sequences.pkl")
    if imitation_data.exists():
        print("Loading imitation data as replay buffer seed...")
        with open(imitation_data, "rb") as f:
            imitation_seqs = pickle.load(f)
        # convert 2-tuple to 3-tuple if needed (add masks)
        for seq in imitation_seqs:
            if len(seq) == 2:
                obs_seq, act_seq = seq
                # generate masks (all True for valid actions)
                masks = np.ones((len(act_seq), 10), dtype=np.float32)
                all_sequences.append((obs_seq, act_seq, masks))
            else:
                all_sequences.append(seq)
        print(f"  Loaded {len(all_sequences)} imitation sequences")

    tc = TypeChart.load()
    data = DataStore()

    # Rust engine setup
    rs_data = None
    rust_evaluator = None
    if args.rust:
        try:
            import crystal_engine_rs as ce
            from training.rust_search_agent import play_search_games as rust_play_search_games
            from training.mcts_evaluator import MctsEvaluator
            rs_data = ce.DataStore(str(Path(__file__).parent.parent / "data"))
            rust_evaluator = MctsEvaluator(args.policy, device=args.device)
            print(f"Rust engine loaded (depth={args.depth}, NN eval)")
        except ImportError:
            print("WARNING: crystal_engine_rs not available, falling back to Python")
            args.rust = False

    # extract initial weights
    weights_path = f"{args.policy}_weights.pt"
    if not Path(weights_path).exists():
        print("Extracting initial model weights...")
        save_model_weights(args.policy, weights_path, device=args.device)

    # load opponent model
    opp_model = OpponentPredictor()
    opp_model.load_state_dict(
        torch.load(args.opp_model, map_location=args.device, weights_only=True))
    opp_model.eval()

    # opponent policy for Rust game driver
    if args.rust:
        smart_opp = SmartAgent(tc, seed=0)
        maxdmg_opp = MaxDamageAgent(tc)
        def mixed_opponent_policy(my_state, opp_state):
            """50/50 Smart/MaxDmg mix (matches existing training)."""
            if random.random() < 0.5:
                return smart_opp.act(my_state, opp_state)
            return maxdmg_opp.act(my_state, opp_state)

    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")

        # ---- Phase 1: Play games with search ----
        depth_str = f"{args.depth}-ply {'Rust' if args.rust else 'Python'}"
        print(f"\n  Phase 1: Playing {args.games_per_iter} games with {depth_str} search...")

        import time
        t0 = time.time()

        if args.rust:
            sequences = rust_play_search_games(
                opponent_policy=mixed_opponent_policy,
                data=data, tc=tc, rs_data=rs_data,
                n_games=args.games_per_iter,
                seed=iteration * 10000,
                depth=args.depth,
                opp_model=opp_model,
                evaluator=rust_evaluator,
            )
        else:
            policy = MaskableRecurrentPPO.load(args.policy, device=args.device)
            agent = LookaheadAgent(policy, opp_model, tc, depth=args.depth, device=args.device)
            sequences = play_search_games(
                agent, data, tc,
                n_games=args.games_per_iter,
                seed=iteration * 10000,
            )

        gen_time = time.time() - t0
        total_steps = sum(len(s[0]) for s in sequences)
        print(f"    Generated {len(sequences)} sequences, {total_steps} steps in {gen_time:.1f}s")

        # ---- Phase 2: Train policy on search data + replay buffer ----
        # combine new search data with accumulated history to prevent forgetting
        all_sequences.extend(sequences)
        # keep a rolling window of recent data
        max_buffer = args.games_per_iter * 5
        if len(all_sequences) > max_buffer:
            all_sequences = all_sequences[-max_buffer:]

        print(f"\n  Phase 2: Training for {args.epochs_per_iter} epochs "
              f"({len(all_sequences)} total sequences)...")
        train_on_sequences(
            args.policy, all_sequences,
            epochs=args.epochs_per_iter,
            lr=args.lr,
            device=args.device,
        )

        # reload updated weights into PPO model
        from training.imitation import load_seq_pretrained_into_ppo
        load_seq_pretrained_into_ppo(
            weights_path, save_path=args.policy, device=args.device,
        )

        # reload evaluator with updated weights for next iteration
        if args.rust and rust_evaluator is not None:
            rust_evaluator = MctsEvaluator(args.policy, device=args.device)

        # ---- Phase 3: Evaluate ----
        print(f"\n  Phase 3: Evaluating...")
        evaluate_current(args.policy, args.opp_model, tc, device=args.device)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
