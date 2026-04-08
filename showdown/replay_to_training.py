#!/usr/bin/env python3
# convert Showdown replay logs into policy training data
# extracts per-turn (features, action) pairs from human games
#
# Usage:
#   .venv/bin/python showdown/replay_to_training.py --out human_training_data.pkl

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from showdown.policy_train import (
    TYPES, TYPE_IDX, ITEMS, ITEM_IDX, STATUSES, STATUS_IDX,
    POKEMON_FEATURES, SIDE_EXTRAS, SIDE_FEATURES, STATE_FEATURES, N_ACTIONS,
)

REPLAY_DIR = Path(__file__).parent / "replays"

# ============================================================
# POKEDEX (gen2 base stats for feature extraction)
# ============================================================

# load from poke-env if available, else use a minimal fallback
try:
    from poke_env.data.gen_data import GenData
    _gd = GenData.from_gen(2)
    POKEDEX = _gd.pokedex
except ImportError:
    POKEDEX = {}

# common gen2 items by species (from chaos stats / competitive knowledge)
COMMON_ITEMS = {
    "snorlax": "leftovers", "zapdos": "leftovers", "tyranitar": "leftovers",
    "skarmory": "leftovers", "cloyster": "leftovers", "starmie": "leftovers",
    "gengar": "leftovers", "exeggutor": "leftovers", "machamp": "leftovers",
    "raikou": "leftovers", "suicune": "leftovers", "forretress": "leftovers",
    "nidoking": "leftovers", "heracross": "leftovers", "vaporeon": "leftovers",
    "jynx": "leftovers", "alakazam": "leftovers", "espeon": "leftovers",
    "umbreon": "leftovers", "blissey": "leftovers", "miltank": "leftovers",
    "marowak": "thickclub", "pikachu": "lightball",
}


def normalize_species(name: str) -> str:
    """Normalize species name to pokedex key format."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def get_base_stats(species: str) -> dict:
    """Get base stats from pokedex."""
    key = normalize_species(species)
    entry = POKEDEX.get(key, {})
    return entry.get("baseStats", {
        "hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80,
    })


def get_types(species: str) -> tuple[str, str]:
    """Get types from pokedex."""
    key = normalize_species(species)
    entry = POKEDEX.get(key, {})
    types = entry.get("types", ["Normal"])
    t1 = types[0].upper()
    t2 = types[1].upper() if len(types) > 1 else "TYPELESS"
    return t1, t2


def calc_stat(base: int, is_hp: bool = False) -> int:
    """Calculate gen2 stat at level 100 with max DVs/EVs."""
    core = ((base + 15) * 2 + 64) * 100 // 100
    return core + 110 if is_hp else core + 5


# ============================================================
# REPLAY PARSER
# ============================================================

class PokemonState:
    """Track state of a single pokemon during replay."""
    def __init__(self, species: str):
        self.species = species
        self.hp_pct = 100.0
        self.alive = True
        self.status = "NONE"
        self.known_moves = []  # moves used so far (up to 4)
        self.item = COMMON_ITEMS.get(normalize_species(species), "leftovers")

    def add_move(self, move_name: str):
        norm = normalize_move(move_name)
        if norm not in self.known_moves and len(self.known_moves) < 4:
            self.known_moves.append(norm)


def normalize_move(move_name: str) -> str:
    """Normalize Showdown move name to engine format."""
    return re.sub(r'[^a-z0-9]', '', move_name.lower())


class SideState:
    """Track state of one side during replay."""
    def __init__(self):
        self.pokemon = {}  # species -> PokemonState
        self.active = None  # current active species
        self.team_order = []  # order pokemon were revealed
        self.spikes = 0
        self.reflect = False
        self.light_screen = False
        self.boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0,
                       "accuracy": 0, "evasion": 0}

    def switch_in(self, species: str, hp_pct: float):
        if species not in self.pokemon:
            self.pokemon[species] = PokemonState(species)
            self.team_order.append(species)
        self.pokemon[species].hp_pct = hp_pct
        self.active = species
        # boosts reset on switch
        self.boosts = {k: 0 for k in self.boosts}

    def get_active(self) -> PokemonState | None:
        if self.active and self.active in self.pokemon:
            return self.pokemon[self.active]
        return None


def pokemon_features_from_state(pstate: PokemonState) -> np.ndarray:
    """Extract 47-dim feature vector from a PokemonState."""
    features = np.zeros(POKEMON_FEATURES, dtype=np.float32)
    i = 0

    # hp fraction
    features[i] = pstate.hp_pct / 100.0
    i += 1
    # alive
    features[i] = 1.0 if pstate.alive and pstate.hp_pct > 0 else 0.0
    i += 1

    # types (18 dims)
    t1, t2 = get_types(pstate.species)
    for t_name in [t.upper() for t in ["bug", "dark", "dragon", "electric", "fighting",
                    "fire", "flying", "ghost", "grass", "ground", "ice", "normal",
                    "poison", "psychic", "rock", "steel", "water", "typeless"]]:
        features[i] = 1.0 if t1 == t_name or t2 == t_name else 0.0
        i += 1

    # stats / 500
    bs = get_base_stats(pstate.species)
    features[i] = calc_stat(bs.get("atk", 80)) / 500.0; i += 1
    features[i] = calc_stat(bs.get("def", 80)) / 500.0; i += 1
    features[i] = calc_stat(bs.get("spa", 80)) / 500.0; i += 1
    features[i] = calc_stat(bs.get("spd", 80)) / 500.0; i += 1
    features[i] = calc_stat(bs.get("spe", 80)) / 500.0; i += 1

    # status (7 dims)
    for s in ["NONE", "BURN", "SLEEP", "FREEZE", "PARALYZE", "POISON", "TOXIC"]:
        features[i] = 1.0 if pstate.status == s else 0.0
        i += 1

    # item (11 dims)
    item_names = ["LEFTOVERS", "THICKCLUB", "LIGHTBALL", "MIRACLEBERRY", "MINTBERRY",
                  "CHARCOAL", "MYSTICWATER", "MAGNET", "NEVERMELTICE", "SCOPELENS", "NONE"]
    item_upper = pstate.item.upper()
    matched = False
    for iname in item_names:
        if item_upper == iname and iname != "NONE":
            features[i] = 1.0
            matched = True
        i += 1
    if not matched:
        features[i - 1] = 1.0  # NONE slot

    # pp fractions (4 moves) -- assume full PP for human games
    for _ in range(4):
        features[i] = 1.0 if len(pstate.known_moves) > 0 else 0.0
        i += 1

    return features


def side_features_from_state(side: SideState) -> np.ndarray:
    """Extract 292-dim feature vector from a SideState."""
    features = np.zeros(SIDE_FEATURES, dtype=np.float32)

    # pokemon features (6 slots)
    for slot_idx in range(6):
        start = slot_idx * POKEMON_FEATURES
        if slot_idx < len(side.team_order):
            species = side.team_order[slot_idx]
            pstate = side.pokemon[species]
            features[start:start + POKEMON_FEATURES] = pokemon_features_from_state(pstate)
        # else: zeros (unrevealed)

    # extras (10 dims)
    e = 6 * POKEMON_FEATURES
    features[e] = side.boosts["atk"] / 6.0
    features[e + 1] = side.boosts["def"] / 6.0
    features[e + 2] = side.boosts["spa"] / 6.0
    features[e + 3] = side.boosts["spd"] / 6.0
    features[e + 4] = side.boosts["spe"] / 6.0
    features[e + 5] = side.boosts["accuracy"] / 6.0
    features[e + 6] = side.boosts["evasion"] / 6.0
    features[e + 7] = side.spikes / 3.0
    features[e + 8] = 1.0 if side.reflect else 0.0
    features[e + 9] = 1.0 if side.light_screen else 0.0

    return features


def state_features_from_sides(p1: SideState, p2: SideState,
                               weather: str) -> np.ndarray:
    """Extract 587-dim feature vector from game state."""
    features = np.zeros(STATE_FEATURES, dtype=np.float32)

    features[0:SIDE_FEATURES] = side_features_from_state(p1)
    features[SIDE_FEATURES:2 * SIDE_FEATURES] = side_features_from_state(p2)

    # weather (3 dims: sun/rain/sand)
    w = 2 * SIDE_FEATURES
    if "sun" in weather.lower():
        features[w] = 1.0
    elif "rain" in weather.lower():
        features[w + 1] = 1.0
    elif "sand" in weather.lower():
        features[w + 2] = 1.0

    return features


def action_to_index(action: str, side: SideState) -> int | None:
    """Map a human action to our 9-dim action index.

    0-3: moves (by position in the active pokemon's known moveset)
    4-8: switches (by bench position)
    Returns None if action can't be mapped.
    """
    if action.startswith("switch:"):
        target_species = action[7:]
        # find bench index: which revealed-but-not-active pokemon is this?
        bench = [s for s in side.team_order if s != side.active]
        for i, sp in enumerate(bench):
            if sp.lower() == target_species.lower() or \
               normalize_species(sp) == normalize_species(target_species):
                if i < 5:
                    return 4 + i
        return None
    else:
        # it's a move -- find in active pokemon's known moves
        active = side.get_active()
        if not active:
            return None
        norm = normalize_move(action)
        for i, m in enumerate(active.known_moves):
            if m == norm:
                return i
        # move not yet known -- add it and return its index
        if len(active.known_moves) < 4:
            active.add_move(action)
            return len(active.known_moves) - 1
        return None


STATUS_MAP = {
    "brn": "BURN", "slp": "SLEEP", "frz": "FREEZE",
    "par": "PARALYZE", "psn": "POISON", "tox": "TOXIC",
}

BOOST_MAP = {
    "atk": "atk", "def": "def", "spa": "spa", "spd": "spd",
    "spe": "spe", "accuracy": "accuracy", "evasion": "evasion",
}


def parse_hp(hp_str: str) -> float:
    """Parse HP string like '75/100' or '0 fnt' to percentage."""
    if "fnt" in hp_str:
        return 0.0
    parts = hp_str.split("/")
    if len(parts) == 2:
        try:
            return float(parts[0]) / float(parts[1]) * 100.0
        except ValueError:
            return float(parts[0])  # already percentage
    try:
        return float(hp_str)
    except ValueError:
        return 100.0


def process_replay(log: str, side: str = "p1") -> list[tuple[np.ndarray, int]]:
    """Process a replay log into (features, action_index) training pairs.

    Args:
        log: raw replay log string
        side: which side to learn from ("p1" or "p2")

    Returns:
        list of (587-dim features, action index 0-8) pairs
    """
    p1 = SideState()
    p2 = SideState()
    weather = "none"
    current_turn = 0
    samples = []

    # snapshot state at turn start for feature extraction
    # action index is computed against the pre-action state
    p1_action = None
    p2_action = None
    p1_action_idx = None
    p2_action_idx = None
    p1_snap = None
    p2_snap = None
    weather_snap = "none"

    lines = log.split("\n")

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        # ---- turn boundary ----
        if cmd == "turn":
            # emit sample for the side we're learning from
            if current_turn > 0:
                if side == "p1" and p1_action_idx is not None and p1_snap is not None:
                    samples.append((p1_snap, p1_action_idx))
                elif side == "p2" and p2_action_idx is not None and p2_snap is not None:
                    samples.append((p2_snap, p2_action_idx))

            current_turn = int(parts[2])
            # snapshot state at turn start
            if side == "p1":
                p1_snap = state_features_from_sides(p1, p2, weather)
            else:
                p2_snap = state_features_from_sides(p2, p1, weather)
            p1_action = None
            p2_action = None
            p1_action_idx = None
            p2_action_idx = None
            weather_snap = weather

        # ---- switches ----
        elif cmd in ("switch", "drag"):
            if len(parts) >= 5:
                slot = parts[2].strip()
                species = parts[3].split(",")[0].strip()
                hp_pct = parse_hp(parts[4].strip())

                if slot.startswith("p1"):
                    if current_turn > 0 and p1_action is None:
                        # compute index BEFORE updating state
                        p1_action = f"switch:{species}"
                        p1_action_idx = action_to_index(p1_action, p1)
                    p1.switch_in(species, hp_pct)
                elif slot.startswith("p2"):
                    if current_turn > 0 and p2_action is None:
                        p2_action = f"switch:{species}"
                        p2_action_idx = action_to_index(p2_action, p2)
                    p2.switch_in(species, hp_pct)

        # ---- moves ----
        elif cmd == "move":
            if len(parts) >= 4:
                slot = parts[2].strip()
                move_name = parts[3].strip()

                if slot.startswith("p1"):
                    p1_action = move_name
                    active = p1.get_active()
                    if active:
                        active.add_move(move_name)
                    p1_action_idx = action_to_index(move_name, p1)
                elif slot.startswith("p2"):
                    p2_action = move_name
                    active = p2.get_active()
                    if active:
                        active.add_move(move_name)
                    p2_action_idx = action_to_index(move_name, p2)

        # ---- damage ----
        elif cmd == "-damage" or cmd == "-heal":
            if len(parts) >= 4:
                slot = parts[2].strip()
                hp_pct = parse_hp(parts[3].strip())

                if slot.startswith("p1"):
                    active = p1.get_active()
                    if active:
                        active.hp_pct = hp_pct
                        if hp_pct <= 0:
                            active.alive = False
                elif slot.startswith("p2"):
                    active = p2.get_active()
                    if active:
                        active.hp_pct = hp_pct
                        if hp_pct <= 0:
                            active.alive = False

        # ---- status ----
        elif cmd == "-status":
            if len(parts) >= 4:
                slot = parts[2].strip()
                status = STATUS_MAP.get(parts[3].strip(), "NONE")
                if slot.startswith("p1"):
                    active = p1.get_active()
                    if active:
                        active.status = status
                elif slot.startswith("p2"):
                    active = p2.get_active()
                    if active:
                        active.status = status

        elif cmd == "-curestatus":
            if len(parts) >= 3:
                slot = parts[2].strip()
                if slot.startswith("p1"):
                    active = p1.get_active()
                    if active:
                        active.status = "NONE"
                elif slot.startswith("p2"):
                    active = p2.get_active()
                    if active:
                        active.status = "NONE"

        # ---- boosts ----
        elif cmd in ("-boost", "-unboost"):
            if len(parts) >= 5:
                slot = parts[2].strip()
                stat = parts[3].strip()
                amount = int(parts[4].strip())
                if cmd == "-unboost":
                    amount = -amount

                stat_key = BOOST_MAP.get(stat)
                if stat_key:
                    if slot.startswith("p1"):
                        p1.boosts[stat_key] = max(-6, min(6,
                            p1.boosts[stat_key] + amount))
                    elif slot.startswith("p2"):
                        p2.boosts[stat_key] = max(-6, min(6,
                            p2.boosts[stat_key] + amount))

        # ---- hazards ----
        elif cmd == "-sidestart":
            if len(parts) >= 4:
                side_ref = parts[2].strip()
                condition = parts[3].strip().lower()
                if "spikes" in condition:
                    if side_ref.startswith("p1"):
                        p1.spikes = min(3, p1.spikes + 1)
                    elif side_ref.startswith("p2"):
                        p2.spikes = min(3, p2.spikes + 1)
                elif "reflect" in condition:
                    if side_ref.startswith("p1"):
                        p1.reflect = True
                    elif side_ref.startswith("p2"):
                        p2.reflect = True
                elif "light screen" in condition:
                    if side_ref.startswith("p1"):
                        p1.light_screen = True
                    elif side_ref.startswith("p2"):
                        p2.light_screen = True

        elif cmd == "-sideend":
            if len(parts) >= 4:
                side_ref = parts[2].strip()
                condition = parts[3].strip().lower()
                if "reflect" in condition:
                    if side_ref.startswith("p1"):
                        p1.reflect = False
                    elif side_ref.startswith("p2"):
                        p2.reflect = False
                elif "light screen" in condition:
                    if side_ref.startswith("p1"):
                        p1.light_screen = False
                    elif side_ref.startswith("p2"):
                        p2.light_screen = False

        # ---- weather ----
        elif cmd == "-weather":
            if len(parts) >= 3:
                w = parts[2].strip().lower()
                if "sun" in w:
                    weather = "sun"
                elif "rain" in w:
                    weather = "rain"
                elif "sand" in w:
                    weather = "sand"
                elif "none" in w:
                    weather = "none"

        # ---- faint ----
        elif cmd == "faint":
            if len(parts) >= 3:
                slot = parts[2].strip()
                if slot.startswith("p1"):
                    active = p1.get_active()
                    if active:
                        active.hp_pct = 0
                        active.alive = False
                elif slot.startswith("p2"):
                    active = p2.get_active()
                    if active:
                        active.hp_pct = 0
                        active.alive = False

    # emit last turn
    if current_turn > 0:
        if side == "p1" and p1_action_idx is not None and p1_snap is not None:
            samples.append((p1_snap, p1_action_idx))
        elif side == "p2" and p2_action_idx is not None and p2_snap is not None:
            samples.append((p2_snap, p2_action_idx))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert replays to training data")
    parser.add_argument("--out", type=str, default="human_training_data.pkl")
    parser.add_argument("--replay-dir", type=str, default=str(REPLAY_DIR))
    parser.add_argument("--min-rating", type=int, default=0,
                        help="Only use replays where at least one player >= this rating")
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    files = sorted(replay_dir.glob("*.json"))
    print(f"Processing {len(files)} replay files...")

    all_features = []
    all_actions = []
    skipped = 0

    for fi, f in enumerate(files):
        with open(f) as fp:
            data = json.load(fp)

        # rating filter
        rating = data.get("rating")
        if args.min_rating > 0 and (not rating or rating < args.min_rating):
            # still allow tournament replays
            if "smogtours" not in data.get("id", ""):
                skipped += 1
                continue

        log = data.get("log", "")
        if not log:
            continue

        # learn from both sides
        for side in ("p1", "p2"):
            samples = process_replay(log, side=side)
            for feat, action in samples:
                all_features.append(feat)
                all_actions.append(action)

        if (fi + 1) % 200 == 0:
            print(f"  {fi + 1}/{len(files)}: {len(all_features)} samples so far")

    features = np.array(all_features, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    # convert hard labels to soft targets (one-hot with label smoothing)
    smoothing = 0.1
    n_classes = N_ACTIONS
    policies = np.full((len(actions), n_classes), smoothing / n_classes,
                       dtype=np.float32)
    for i, a in enumerate(actions):
        policies[i, a] += (1.0 - smoothing)

    print(f"\n{len(features)} total samples from {len(files) - skipped} replays "
          f"(skipped {skipped})")
    print(f"  state_dim={features.shape[1]}, action_dim={policies.shape[1]}")

    # action distribution
    counts = np.bincount(actions, minlength=N_ACTIONS)
    labels = ["move0", "move1", "move2", "move3",
              "switch1", "switch2", "switch3", "switch4", "switch5"]
    print("  action distribution:")
    for l, c in zip(labels, counts):
        print(f"    {l}: {c} ({c / len(actions) * 100:.1f}%)")

    # save in same format as MCTS data for the trainer
    # (features, soft_policies) tuple
    with open(args.out, "wb") as f:
        pickle.dump((features, policies), f)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
