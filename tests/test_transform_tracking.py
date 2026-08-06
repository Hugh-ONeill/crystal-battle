# Transformed-Imposter-Ditto live rebuild. poke-env tracks a transform's
# temporary fields (types/ability/base-stats/boosts/revealed moves) but keeps
# mon.species and the request stats as base Ditto — and its parse_request
# ASSERTS on the first post-transform request, because the server names all
# four copied moves while poke-env only knows the target's revealed ones.
# These tests pin the translator's transform record, the tolerant
# parse_request patch, and the rebuilt engine mon.

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator


def _req_mon(ident, details, condition, active, stats, moves, ability, item):
    return {
        "ident": ident, "details": details, "condition": condition,
        "active": active, "stats": stats, "moves": moves,
        "baseAbility": ability, "ability": ability, "item": item,
        "pokeball": "pokeball",
    }


DITTO = _req_mon("p1: Ditto", "Ditto, L100", "269/269", True,
                 {"atk": 96, "def": 96, "spa": 96, "spd": 96, "spe": 132},
                 ["transform"], "imposter", "choicescarf")

REQ_BASE = {
    "active": [{"moves": [
        {"move": "Transform", "id": "transform", "pp": 16, "maxpp": 16,
         "target": "normal", "disabled": False},
    ]}],
    "side": {"name": "wizbot", "id": "p1", "pokemon": [DITTO]},
}

# after Imposter fires, the server names the copied moveset to the owner —
# including the moves the target never revealed (outrage/sd/rock here)
REQ_TRANSFORMED = {
    "active": [{"moves": [
        {"move": "Earthquake", "id": "earthquake", "pp": 5, "maxpp": 5,
         "target": "normal", "disabled": False},
        {"move": "Outrage", "id": "outrage", "pp": 5, "maxpp": 5,
         "target": "normal", "disabled": False},
        {"move": "Swords Dance", "id": "swordsdance", "pp": 5, "maxpp": 5,
         "target": "normal", "disabled": False},
        {"move": "Stealth Rock", "id": "stealthrock", "pp": 5, "maxpp": 5,
         "target": "normal", "disabled": False},
    ]}],
    "side": {"name": "wizbot", "id": "p1", "pokemon": [DITTO]},
}

PROTOCOL = [
    ["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"],
    ["", "-boost", "p2a: Garchomp", "atk", "2"],
    ["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ditto"],
    ["", "switch", "p1a: Ditto", "Ditto, L100", "269/269"],
    ["", "-transform", "p1a: Ditto", "p2a: Garchomp",
     "[from] ability: Imposter"],
    ["", "turn", "1"],
]


def make_transformed_battle():
    b = Battle("battle-gen9ou-tftest-1", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQ_BASE)
    for msg in PROTOCOL:
        b.parse_message(msg)
    b.parse_request(REQ_TRANSFORMED)   # exercises the tolerant patch

    tr = Gen9Translator()
    tr.observe_events(PROTOCOL, role="p1", turn=0)
    return tr, b


def test_parse_request_survives_copied_unrevealed_moves():
    # without the patch this parse_request raises AssertionError on
    # "outrage" (copied but never revealed by the target)
    _, b = make_transformed_battle()
    assert sorted(m.id for m in b.available_moves) == \
        ["earthquake", "outrage", "stealthrock", "swordsdance"]


def test_parse_request_patch_is_signature_transparent():
    # the live player calls parse_request(request, strict) POSITIONALLY —
    # the patch must pass extra args through (a (self, request)-pinned
    # wrapper crashed every live request while offline tests stayed green)
    b = Battle("battle-gen9ou-tftest-3", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQ_BASE, False)
    for msg in PROTOCOL:
        b.parse_message(msg)
    b.parse_request(REQ_TRANSFORMED, False)
    assert sorted(m.id for m in b.available_moves) == \
        ["earthquake", "outrage", "stealthrock", "swordsdance"]


def test_transform_record_set_and_cleared_on_slot_change():
    tr, _ = make_transformed_battle()
    assert tr._transform["me"] == {"who": "ditto", "into": "garchomp"}
    tr.observe_events(
        [["", "switch", "p1a: Ninetales", "Ninetales, L100, F", "100/100"]],
        role="p1", turn=2)
    assert tr._transform["me"] is None


def test_transformed_ditto_rebuild():
    tr, b = make_transformed_battle()
    state = tr.translate(b)
    p = state.side_one.pokemon[0]
    # copied half: identity, typing, a Garchomp-class statline
    assert p.id == "garchomp"
    assert tuple(p.types) == ("dragon", "ground")
    assert p.base_ability == "imposter"      # keeps the engine revert live
    assert p.speed >= 220                    # Garchomp spread, not Ditto 132
    # own half: HP bar and item stay Ditto's
    assert p.hp == 269 and p.maxhp == 269
    assert p.item == "choicescarf"
    # moves: the request's copied set, 5 PP each, Transform itself gone
    ids = [m.id for m in p.moves]
    assert ids == ["earthquake", "outrage", "swordsdance", "stealthrock"]
    assert all(m.pp == 5 for m in p.moves if m.id != "none")
    # the copied +2 atk rides on the side via poke-env's own boost copy
    assert state.side_one.attack_boost == 2


def test_untransformed_ditto_builds_normally():
    b = Battle("battle-gen9ou-tftest-2", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQ_BASE)
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M",
                     "100/100"])
    b.parse_message(["", "switch", "p1a: Ditto", "Ditto, L100", "269/269"])
    b.parse_message(["", "turn", "1"])
    tr = Gen9Translator()
    state = tr.translate(b)
    p = state.side_one.pokemon[0]
    assert p.id == "ditto"
    assert p.speed == 132
    assert [m.id for m in p.moves][0] == "transform"
