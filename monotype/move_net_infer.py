"""
Inference helpers for the MoveNet — bridges live engine state to net input.

Given a poke-engine Side, identifies:
  - the active mon
  - the team's monotype
  - the active's canonical 4 moves (from Smogon stats)
then runs the net and returns {move_id: prob} for the actor's 4 candidate moves.

The intended use is symmetric to chaos priors: call once per turn per side
to get a state-conditional move-prob distribution; use it as a diagnostic
in trace_match or as a re-weighting layer on MCTS visit counts in bench.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from monotype.canonical_sets import build_canonical_sets, MONOTYPE_TYPES
from monotype.chaos_priors import _detect_side_type
from monotype.extract_move_training_data import extract_moves_from_block
from monotype.featurizer_lead_preview import featurize_team_from_state_side
from monotype.featurizer_move_state import (
    BOOST_STATS, TurnState, featurize_turn_state,
    normalize_status, normalize_terrain, normalize_weather,
)
from monotype.move_net import MoveNet, MoveNetV2


def _turn_state_from_engine(state) -> TurnState:
    """Build a TurnState snapshot from a live poke_engine State object.

    Maps engine attribute names to the featurizer's expected field set.
    Volatile / form-change information isn't in the V2 featurizer, so
    those fields are left at defaults.
    """
    ts = TurnState()
    # Boosts (BOOST_STATS = atk/def/spa/spd/spe)
    eng_attr = {
        "atk": "attack_boost", "def": "defense_boost",
        "spa": "special_attack_boost", "spd": "special_defense_boost",
        "spe": "speed_boost",
    }
    for stat in BOOST_STATS:
        ts.boosts_p1[stat] = getattr(state.side_one, eng_attr[stat])
        ts.boosts_p2[stat] = getattr(state.side_two, eng_attr[stat])

    # Status on actives only (per featurizer convention)
    a1 = state.side_one.pokemon[int(state.side_one.active_index)]
    a2 = state.side_two.pokemon[int(state.side_two.active_index)]
    ts.status_p1 = normalize_status((a1.status or "none").lower())
    ts.status_p2 = normalize_status((a2.status or "none").lower())

    # Field
    weather_id = (state.weather or "none").lower()
    # poke-engine weather ids: "sun", "rain", "sand", "snow", etc.
    # The featurizer's normalize_weather() expects Showdown move names; do a
    # quick remap so engine-side ids land on the right one-hot slot.
    weather_remap = {
        "sun": "sun", "rain": "rain", "snow": "snow", "sand": "sand",
        "harshsun": "harshsun", "heavyrain": "heavyrain", "none": "none",
    }
    ts.weather = weather_remap.get(weather_id, "none")
    terrain_id = (state.terrain or "none").lower()
    terrain_remap = {
        "electricterrain": "electric", "grassyterrain": "grassy",
        "psychicterrain": "psychic", "mistyterrain": "misty", "none": "none",
    }
    ts.terrain = terrain_remap.get(terrain_id, "none")
    ts.trick_room = bool(state.trick_room)

    # Hazards / screens / tailwind from side_conditions
    for side_obj, prefix in [(state.side_one, "1"), (state.side_two, "2")]:
        sc = side_obj.side_conditions
        setattr(ts, f"sr_p{prefix}", bool(getattr(sc, "stealth_rock", 0)))
        setattr(ts, f"spikes_p{prefix}", int(getattr(sc, "spikes", 0) or 0))
        setattr(ts, f"tspikes_p{prefix}", int(getattr(sc, "toxic_spikes", 0) or 0))
        setattr(ts, f"web_p{prefix}", bool(getattr(sc, "sticky_web", 0)))
        setattr(ts, f"reflect_p{prefix}", bool(getattr(sc, "reflect", 0)))
        setattr(ts, f"lightscreen_p{prefix}", bool(getattr(sc, "light_screen", 0)))
        setattr(ts, f"auroraveil_p{prefix}", bool(getattr(sc, "aurora_veil", 0)))
        if prefix == "1":
            ts.tailwind_p1 = bool(getattr(sc, "tailwind", 0))
        else:
            ts.tailwind_p2 = bool(getattr(sc, "tailwind", 0))
    return ts


TYPE_NAME_TO_IDX = {t: i for i, t in enumerate(MONOTYPE_TYPES)}


_CACHE: dict[str, dict] = {}


def load_move_net(path: str | Path, device: str = "cpu") -> dict:
    """Load and cache a trained MoveNet checkpoint by path string.

    Auto-detects V1 vs V2 by the presence of `state_encoder.0.weight`'s
    input dim (V1 = MON_DIM*2+34 = 136; V2 includes 53-dim turn state).
    """
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    in_dim = sd["state_encoder.0.weight"].shape[1]
    # V1: 51*2 + 2 + 2*16 = 136. V2: V1 + 53 = 189.
    is_v2 = in_dim >= 189
    cls = MoveNetV2 if is_v2 else MoveNet
    net = cls(n_moves=ckpt["n_moves"]).to(device)
    net.load_state_dict(sd)
    net.eval()
    vocab = list(ckpt["vocab"])
    bundle = {
        "net": net,
        "is_v2": is_v2,
        "vocab": vocab,
        "move_to_idx": {m: i for i, m in enumerate(vocab)},
        "n_moves": ckpt["n_moves"],
        "device": device,
        "canonical_sets": build_canonical_sets(),
    }
    _CACHE[key] = bundle
    return bundle


def predict_active_move_probs(state, actor_side, bundle: dict) -> dict[str, float]:
    """Return {move_id: prob} for the actor side's currently active mon.

    Returns empty dict if the active species lacks a canonical set or the
    side's monotype can't be detected.
    """
    if actor_side is state.side_one:
        opp_side = state.side_two
    else:
        opp_side = state.side_one

    species_actor = tuple(p.id for p in actor_side.pokemon)
    species_opp = tuple(p.id for p in opp_side.pokemon)
    type_actor = _detect_side_type(species_actor)
    type_opp = _detect_side_type(species_opp)
    if not type_actor or not type_opp:
        return {}

    # Canonical 4-move set for the actor's active
    cs = bundle["canonical_sets"].get(type_actor, {})
    active = actor_side.pokemon[int(actor_side.active_index)]
    # canonical_sets keys use display-case species names ("Heatran", "Goodra-Hisui");
    # poke-engine ids are lowercase no-space ("heatran", "goodrahisui"). Match by norm.
    norm_id = active.id.replace(" ", "").replace("-", "").lower()
    block = None
    for sp_name, sp_block in cs.items():
        if sp_name.replace("-", "").replace(" ", "").lower() == norm_id:
            block = sp_block
            break
    if block is None:
        # Try base form (e.g. "Ogerpon" for "ogerponhearthflame")
        base = norm_id.split("-")[0]
        for sp_name, sp_block in cs.items():
            if sp_name.split("-")[0].lower() == base:
                block = sp_block
                break
    if block is None:
        return {}
    moves = extract_moves_from_block(block)
    if len(moves) < 4:
        return {}

    # Featurize the actor's active and opp's active (re-using lead-preview featurizer)
    actor_feats_all = featurize_team_from_state_side(actor_side)
    opp_feats_all = featurize_team_from_state_side(opp_side)
    actor_idx = int(actor_side.active_index)
    opp_idx = int(opp_side.active_index)

    move_to_idx = bundle["move_to_idx"]
    n_moves = bundle["n_moves"]
    cand = [move_to_idx.get(m, n_moves) for m in moves]

    device = bundle["device"]
    net = bundle["net"]
    is_v2 = bundle.get("is_v2", False)
    actor_t = torch.from_numpy(actor_feats_all[actor_idx]).unsqueeze(0).to(device)
    opp_t = torch.from_numpy(opp_feats_all[opp_idx]).unsqueeze(0).to(device)
    cand_t = torch.tensor([cand], dtype=torch.long, device=device)
    hpa = torch.tensor([active.hp / max(1, active.maxhp)],
                       dtype=torch.float32, device=device)
    hpo = torch.tensor([opp_side.pokemon[opp_idx].hp /
                        max(1, opp_side.pokemon[opp_idx].maxhp)],
                       dtype=torch.float32, device=device)
    tta = torch.tensor([TYPE_NAME_TO_IDX[type_actor]],
                       dtype=torch.long, device=device)
    tto = torch.tensor([TYPE_NAME_TO_IDX[type_opp]],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        if is_v2:
            actor_side_label = "p1" if actor_side is state.side_one else "p2"
            ts = _turn_state_from_engine(state)
            ts_arr = featurize_turn_state(ts, actor_side_label)
            ts_t = torch.from_numpy(ts_arr).unsqueeze(0).to(device)
            logits = net(actor_t, opp_t, ts_t, cand_t, hpa, hpo, tta, tto)
        else:
            logits = net(actor_t, opp_t, cand_t, hpa, hpo, tta, tto)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return {moves[i]: probs[i] for i in range(4)}
