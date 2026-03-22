# Rust-accelerated search agent + game driver for AlphaZero-style training
# Drives games entirely through crystal_engine_rs, only calling Python for:
#   - Team building (gym_env.team_builder)
#   - Observation building (gym_env.obs_builder)
#   - Opponent model inference (training.opponent_model)
#
# Usage:
#   .venv/bin/python training/rust_search_agent.py --games 100

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data_loader import DataStore
from engine.types import TypeChart
from gym_env.obs_builder import build_observation, OBS_SIZE
from gym_env.team_builder import build_team

try:
    import crystal_engine_rs as ce
except ImportError:
    print("ERROR: crystal_engine_rs not found")
    sys.exit(1)

DATA_DIR = str(Path(__file__).parent.parent / "data")


# ============================================================
# OBSERVATION BRIDGE
# ============================================================

def build_obs_from_rust(rs_battle, tc):
    """Build observation array from Rust battle state.

    Bridges Rust state -> Python obs_builder by reading fields
    through PyO3 bindings. This is the only Python-side per-turn cost.
    """
    # obs_builder expects Python PlayerState objects, but we can
    # build a lightweight adapter since it only reads fields
    p1 = _RustPlayerAdapter(rs_battle.p1)
    p2 = _RustPlayerAdapter(rs_battle.p2)
    return build_observation(
        p1, p2, tc, turn=rs_battle.turn,
        weather=rs_battle.weather, weather_turns=rs_battle.weather_turns,
    )


class _RustPlayerAdapter:
    """Lightweight adapter making a Rust PlayerState look like a Python one
    to obs_builder. Only exposes fields that obs_builder reads."""

    def __init__(self, rs_ps):
        self._rs = rs_ps

    @property
    def active(self):
        return _RustPokemonAdapter(self._rs.active)

    @property
    def active_index(self):
        return self._rs.active_index

    @property
    def team(self):
        return [_RustPokemonAdapter(p) for p in self._rs.team]

    @property
    def side(self):
        return self._rs.side

    @property
    def active_turns(self):
        return self._rs.active_turns

    @property
    def alive_count(self):
        return self._rs.alive_count

    @property
    def total_hp_frac(self):
        return self._rs.total_hp_frac()

    @property
    def must_switch(self):
        return self._rs.must_switch

    @property
    def is_defeated(self):
        return self._rs.is_defeated

    def valid_actions(self, opponent=None, type_chart=None):
        """For compatibility with Python opponent policies."""
        from engine.actions import UseMove, Switch, Struggle
        opp = opponent._rs if isinstance(opponent, _RustPlayerAdapter) else None
        mask = self._rs.valid_action_mask(opp, filter_immune=type_chart is not None)
        actions = []
        for i in range(4):
            if mask[i]:
                actions.append(UseMove(slot_index=i))
        for i in range(4, 10):
            if mask[i]:
                actions.append(Switch(team_index=i - 4))
        if not actions:
            actions.append(Struggle())
        return actions

    def valid_action_mask(self, opponent=None, type_chart=None):
        opp = opponent._rs if isinstance(opponent, _RustPlayerAdapter) else None
        return list(self._rs.valid_action_mask(opp, filter_immune=type_chart is not None))


class _RustPokemonAdapter:
    """Makes a Rust Pokemon look like a Python one to obs_builder."""

    def __init__(self, rs_mon):
        self._rs = rs_mon

    @property
    def name(self):
        return self._rs.name

    @property
    def species(self):
        return self  # obs_builder accesses species.types, species.base_stats

    @property
    def types(self):
        return self._rs.types

    @property
    def base_stats(self):
        s = self._rs.stats
        # obs_builder reads base_stats dict -- approximate from computed stats
        # (this is only used for base stat features, not damage calc)
        return {"attack": s[1], "defense": s[2],
                "special_attack": s[3], "special_defense": s[4]}

    @property
    def stats(self):
        s = self._rs.stats
        return {"hp": s[0], "attack": s[1], "defense": s[2],
                "special_attack": s[3], "special_defense": s[4], "speed": s[5]}

    @property
    def current_hp(self):
        return self._rs.current_hp

    @property
    def max_hp(self):
        return self._rs.max_hp

    @property
    def hp_frac(self):
        return self._rs.hp_frac

    @property
    def is_fainted(self):
        return self._rs.is_fainted

    @property
    def status(self):
        return self._rs.status

    @property
    def status_turns(self):
        return self._rs.status_turns

    @property
    def confusion_turns(self):
        return self._rs.confusion_turns

    @property
    def stat_stages(self):
        s = self._rs.stat_stages
        return {"attack": s[0], "defense": s[1], "special_attack": s[2],
                "special_defense": s[3], "speed": s[4], "accuracy": s[5], "evasion": s[6]}

    @property
    def move_slots(self):
        return [_RustMoveSlotAdapter(ms) for ms in self._rs.move_slots]

    @property
    def flinched(self):
        return self._rs.flinched

    @property
    def leech_seeded(self):
        return self._rs.leech_seeded

    @property
    def protected(self):
        return self._rs.protected

    @property
    def protect_consecutive(self):
        return self._rs.protect_consecutive

    @property
    def recharging(self):
        return self._rs.recharging

    def has_any_pp(self):
        return self._rs.has_any_pp()


class _RustMoveSlotAdapter:
    def __init__(self, rs_slot):
        self._rs = rs_slot

    @property
    def template(self):
        return _RustMoveTemplateAdapter(self._rs.template)

    @property
    def current_pp(self):
        return self._rs.current_pp

    @property
    def has_pp(self):
        return self._rs.has_pp


class _RustMoveTemplateAdapter:
    def __init__(self, rs_tmpl):
        self._rs = rs_tmpl

    @property
    def id(self):
        return self._rs.id

    @property
    def name(self):
        return self._rs.name

    @property
    def type(self):
        return self._rs.move_type

    @property
    def power(self):
        return self._rs.power

    @property
    def accuracy(self):
        return self._rs.accuracy

    @property
    def pp(self):
        return self._rs.pp

    @property
    def priority(self):
        return self._rs.priority

    @property
    def damage_class(self):
        return self._rs.damage_class

    @property
    def meta(self):
        m = self._rs.meta
        return {
            "ailment_id": m.ailment_id,
            "min_hits": m.min_hits,
            "max_hits": m.max_hits,
            "drain": m.drain,
            "healing": m.healing,
            "crit_rate": m.crit_rate,
            "ailment_chance": m.ailment_chance,
            "flinch_chance": m.flinch_chance,
            "stat_chance": m.stat_chance,
        }


# ============================================================
# GAME DRIVER
# ============================================================

def play_search_games(
    opponent_policy,
    data: DataStore,
    tc: TypeChart,
    rs_data,
    n_games: int = 500,
    seed: int = 0,
    depth: int = 2,
    n_opp_samples: int = 5,
    opp_model=None,
    evaluator=None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Play games with Rust search and record (obs, action, mask) sequences.

    The game is driven by the Rust engine. Observations built in Rust.
    When evaluator is provided, uses NN leaf evaluation (search_2ply_nn).
    Otherwise falls back to heuristic evaluation (search_2ply).
    """
    # build evaluator callback if NN evaluator is provided
    eval_callback = None
    if evaluator is not None:
        if hasattr(evaluator, 'predict_batch'):
            # feedforward value net: only returns values, no priors
            def eval_callback(obs_batch, mask_batch):
                obs_np = np.array(obs_batch, dtype=np.float32)
                values = evaluator.predict_batch(obs_np)
                # dummy priors (not used by search_2ply_nn aggregation)
                priors = [[0.1] * 10] * len(values)
                return values.tolist() if hasattr(values, 'tolist') else list(values), priors
        else:
            # MctsEvaluator: returns (values, priors)
            def eval_callback(obs_batch, mask_batch):
                obs_np = np.array(obs_batch, dtype=np.float32)
                mask_np = np.array(mask_batch, dtype=np.float32)
                values, priors = evaluator.evaluate_batch(obs_np, mask_np)
                return values, priors

    # build opponent prediction callback if opp model is provided
    opp_predict_callback = None
    if opp_model is not None:
        def opp_predict_callback(obs_batch, mask_batch):
            """Predict opponent actions for a batch of states.
            Returns list of [(action, prob), ...] per state."""
            results = []
            for obs_vec, mask_vec in zip(obs_batch, mask_batch):
                obs_np = np.array(obs_vec, dtype=np.float32)
                probs, _ = opp_model.predict_single(obs_np, None)
                mask_np = np.array(mask_vec, dtype=np.float32)
                probs = probs * mask_np
                s = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs = mask_np / mask_np.sum()
                top = np.argsort(probs)[::-1][:n_opp_samples]
                results.append([(int(i), float(probs[i])) for i in top if probs[i] > 0.01])
                if not results[-1]:
                    results[-1] = [(int(np.argmax(probs)), 1.0)]
            return results

    sequences = []
    rng = random.Random(seed)
    opp_hidden = None

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**31)

        # build teams in Python, convert to Rust (with DVs for HP resolution)
        t1_py = build_team(data, rng=random.Random(game_seed + 100), tier="ou")
        t2_py = build_team(data, rng=random.Random(game_seed + 200), tier="ou")

        rs_t1 = [_py_mon_to_rs(rs_data, m) for m in t1_py]
        rs_t2 = [_py_mon_to_rs(rs_data, m) for m in t2_py]
        rs_battle = ce.create_battle(rs_t1, rs_t2, seed=game_seed + 300)

        opp_hidden = None  # reset per game
        obs_seq = []
        action_seq = []
        mask_seq = []

        for turn in range(100):
            if rs_battle.is_over:
                break

            # build observation in Rust
            obs = np.array(ce.build_observation(rs_battle), dtype=np.float32)
            p1_mask = rs_battle.p1.valid_action_mask(rs_battle.p2, filter_immune=True)
            mask = np.array(p1_mask, dtype=np.float32)
            p1_valid = [i for i in range(10) if p1_mask[i]]

            if not p1_valid:
                break

            # ---- P1: Rust search picks the action ----
            if len(p1_valid) == 1:
                best_action = p1_valid[0]
                if opp_model is not None:
                    _, opp_hidden = opp_model.predict_single(obs, opp_hidden)
            else:
                opp_actions = _get_opp_actions(
                    rs_battle, obs, opp_model, opp_hidden, n_opp_samples,
                )
                if opp_model is not None:
                    _, opp_hidden = opp_model.predict_single(obs, opp_hidden)

                search_seed = game_seed * 1000 + turn

                if depth >= 2 and eval_callback is not None:
                    # 2-ply with NN leaf eval, static opp weights
                    ranked = ce.search_2ply_nn(
                        rs_battle, p1_valid, opp_actions,
                        evaluator_fn=eval_callback,
                        base_seed=search_seed,
                    )
                elif depth >= 2:
                    # 2-ply heuristic
                    ranked = ce.search_2ply(
                        rs_battle, p1_valid, opp_actions, opp_actions,
                        base_seed=search_seed,
                    )
                elif eval_callback is not None:
                    ranked = ce.search_1ply_nn(
                        rs_battle, p1_valid, opp_actions,
                        eval_callback, base_seed=search_seed,
                    )
                else:
                    ranked = ce.search_1ply(
                        rs_battle, p1_valid, opp_actions,
                        base_seed=search_seed,
                    )
                best_action = ranked[0][0] if ranked else p1_valid[0]

            obs_seq.append(obs)
            action_seq.append(best_action)
            mask_seq.append(mask)

            # ---- P2: opponent policy ----
            p2_adapter = _RustPlayerAdapter(rs_battle.p2)
            p1_adapter = _RustPlayerAdapter(rs_battle.p1)
            p2_action = opponent_policy(p2_adapter, p1_adapter)
            a2_int = _py_action_to_int(p2_action)

            # resolve turn in Rust
            rs_battle.resolve_turn(best_action, a2_int)

            # forced switches
            _handle_forced_switches_rs(rs_battle, opponent_policy)

            # record forced switch actions for P1
            if rs_battle.p1.must_switch:
                sw_obs = np.array(ce.build_observation(rs_battle), dtype=np.float32)
                sw_mask = np.array(
                    rs_battle.p1.valid_action_mask(rs_battle.p2, filter_immune=False),
                    dtype=np.float32,
                )
                sw_idx = _first_alive_bench_idx(rs_battle.p1)
                if sw_idx is not None:
                    obs_seq.append(sw_obs)
                    action_seq.append(4 + sw_idx)
                    mask_seq.append(sw_mask)
                    rs_battle.resolve_forced_switches(sw_idx, None)

        # determine game outcome from P1's perspective
        if rs_battle.winner == 1:
            outcome = 1.0
        elif rs_battle.winner == 2:
            outcome = -1.0
        else:
            outcome = 0.0

        if len(obs_seq) >= 3:
            sequences.append((
                np.array(obs_seq, dtype=np.float32),
                np.array(action_seq, dtype=np.int64),
                np.array(mask_seq, dtype=np.float32),
                outcome,
            ))

        if (game_idx + 1) % 100 == 0:
            print(f"    {game_idx + 1}/{n_games} games")

    return sequences


def _py_mon_to_rs(rs_data, mon):
    """Convert a Python Pokemon to Rust, passing DVs for correct HP resolution."""
    move_ids = [s.template.id for s in mon.move_slots]
    dvs = [mon.dvs.get('attack', 15), mon.dvs.get('defense', 15),
           mon.dvs.get('speed', 15), mon.dvs.get('special', 15)]
    return rs_data.build_pokemon(mon.species.id, move_ids, dvs=dvs)


def _get_opp_actions(rs_battle, obs, opp_model, opp_hidden, n_samples):
    """Get opponent action weights for search."""
    p2_mask = rs_battle.p2.valid_action_mask(rs_battle.p1, filter_immune=True)
    p2_valid = [i for i in range(10) if p2_mask[i]]

    if not p2_valid:
        return [(0, 1.0)]

    if opp_model is None:
        w = 1.0 / len(p2_valid)
        return [(a, w) for a in p2_valid[:n_samples]]

    opp_probs, _ = opp_model.predict_single(obs, opp_hidden)
    opp_probs = opp_probs * np.array(p2_mask, dtype=np.float32)
    total = opp_probs.sum()
    if total > 0:
        opp_probs /= total
    else:
        opp_probs = np.array(p2_mask, dtype=np.float32)
        opp_probs /= opp_probs.sum()

    top_idx = np.argsort(opp_probs)[::-1][:n_samples]
    result = [(int(i), float(opp_probs[i])) for i in top_idx if opp_probs[i] > 0.01]
    if not result:
        result = [(int(np.argmax(opp_probs)), 1.0)]
    return result


def _handle_forced_switches_rs(rs_battle, opponent_policy):
    """Handle P2 forced switches through the opponent policy."""
    if rs_battle.p2.must_switch:
        p2_adapter = _RustPlayerAdapter(rs_battle.p2)
        p1_adapter = _RustPlayerAdapter(rs_battle.p1)
        p2_action = opponent_policy(p2_adapter, p1_adapter)
        a2_int = _py_action_to_int(p2_action)
        if 4 <= a2_int <= 9:
            rs_battle.resolve_forced_switches(None, a2_int - 4)
        else:
            # fallback
            sw = _first_alive_bench_idx(rs_battle.p2)
            if sw is not None:
                rs_battle.resolve_forced_switches(None, sw)


def _first_alive_bench_idx(rs_ps):
    for i, p in enumerate(rs_ps.team):
        if i != rs_ps.active_index and not p.is_fainted:
            return i
    return None


def _py_action_to_int(action) -> int:
    from engine.actions import UseMove, Switch, Struggle
    if isinstance(action, UseMove):
        return action.slot_index
    elif isinstance(action, Switch):
        return action.team_index + 4
    elif isinstance(action, Struggle):
        return 10
    return 10


# ============================================================
# STANDALONE TEST
# ============================================================

def main():
    """Quick test: play a few games with Rust 2-ply search vs SmartAgent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    data = DataStore()
    tc = TypeChart.load()
    rs_data = ce.DataStore(DATA_DIR)

    from training.baselines import SmartAgent, MaxDamageAgent

    smart = SmartAgent(tc, seed=42)

    def smart_policy(my_state, opp_state):
        return smart.act(my_state, opp_state)

    print(f"Playing {args.games} games with Rust {args.depth}-ply search vs Smart...")
    t0 = time.time()
    sequences = play_search_games(
        opponent_policy=smart_policy,
        data=data, tc=tc, rs_data=rs_data,
        n_games=args.games, seed=42, depth=args.depth,
    )
    elapsed = time.time() - t0

    total_steps = sum(len(s[0]) for s in sequences)
    print(f"  {len(sequences)} games, {total_steps} steps in {elapsed:.2f}s")
    print(f"  {total_steps / elapsed:.0f} steps/s, {len(sequences) / elapsed:.1f} games/s")


if __name__ == "__main__":
    main()
