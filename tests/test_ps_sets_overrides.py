# The ps_sets meta-tech overlay + slashed-move expansion.
#
# PS's sets export flattens Smogon's slashed move options to the FIRST choice
# (verified 2026-07-30: 0/278 gen9ou sets carry an alternative), so tech moves
# like Kingambit's slashed Low Kick were invisible to world-0 set sampling —
# our own Kingambit switched into theirs reading it as safe when Low Kick is a
# 4x 120-BP punish. ps_sets_overrides.json re-slashes such sets; _parse_set
# expands each combination into its own candidate with the weight split.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.ps_sets import PSSetsIndex


def test_kingambit_low_kick_variant_present():
    idx = PSSetsIndex("gen9ou")
    cands = idx.candidates.get("kingambit", [])
    assert cands, "kingambit missing from the gen9ou index"
    assert any("lowkick" in c["moves"] for c in cands), \
        "the overlay's Low Kick variant did not reach the index"
    # a revealed Low Kick must keep only compatible candidates
    kept = idx.consistent("kingambit", known_moves=("Low Kick",))
    assert kept and all("lowkick" in c["moves"] for c in kept)
    # and the standard Iron Head reveal still resolves
    kept_ih = idx.consistent("kingambit", known_moves=("Iron Head",))
    assert kept_ih and all("ironhead" in c["moves"] for c in kept_ih)


def test_slashed_slots_expand_and_split_weight(tmp_path):
    raw = {"gen9ou": {"dex": {"Testmon": {"Bulky": {
        "moves": ["Tackle", ["Surf", "Flamethrower"], "Recover", "Toxic"],
        "ability": "Torrent", "item": "Leftovers", "nature": "Bold",
        "evs": {"hp": 252}}}}}}
    p = tmp_path / "sets.json"
    p.write_text(json.dumps(raw))
    idx = PSSetsIndex("gen9ou", path=p,
                      base_stats={"testmon": {"spe": 50}}, overrides_path=None)
    cands = idx.candidates["testmon"]
    assert len(cands) == 2
    assert {c["moves"][1] for c in cands} == {"surf", "flamethrower"}
    assert abs(sum(c["weight"] for c in cands) - 1.0) < 1e-9
    assert {c["name"] for c in cands} == {"Bulky (Surf)", "Bulky (Flamethrower)"}
    # revealed-move filtering keeps exactly the matching variant
    kept = idx.consistent("testmon", known_moves=("Flamethrower",))
    assert [c["name"] for c in kept] == ["Bulky (Flamethrower)"]


def test_flat_sets_unchanged_by_expansion(tmp_path):
    raw = {"gen9ou": {"dex": {"Plainmon": {"Std": {
        "moves": ["Tackle", "Recover"], "ability": "Torrent",
        "item": "Leftovers", "nature": "Bold", "evs": {"hp": 252}}}}}}
    p = tmp_path / "sets.json"
    p.write_text(json.dumps(raw))
    idx = PSSetsIndex("gen9ou", path=p,
                      base_stats={"plainmon": {"spe": 50}}, overrides_path=None)
    cands = idx.candidates["plainmon"]
    assert len(cands) == 1
    assert cands[0]["name"] == "Std"
    assert cands[0]["weight"] == 1.0
