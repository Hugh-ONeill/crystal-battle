# Nickname robustness. Protocol IDENTS carry the nickname, never the species
# ('|switch|p2a: Steve|Slowking-Galar, L100, M|100/100'), so every layer must
# read species from DETAILS. The forme bug of 2026-08-04 was the accidental
# version of this (idents give a forme its BASE name, so 'p2a: Slowking' for a
# Slowking-Galar silently missed every upkeep proof); these tests are the
# deliberate version, with nicknames that share nothing with the species.
#
# Scope note: the sim REJECTS both halves of the classic troll — a nickname
# equal to another species ("must not be nicknamed a different Pokemon
# species") and duplicate nicknames within a team ("must have different
# nicknames") — so six mons called Ditto cannot reach us through a validated
# format. Arbitrary DISTINCT nicknames are legal and ordinary, which is what
# is fuzzed here. Our ladder opponents so far are bots that leave nicknames
# at the species default, so this path is otherwise unexercised.

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator
from showdown.set_inference import BattleObservations


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


def _req_mon(ident, details, condition, active, stats, moves, ability, item,
             tera=None):
    d = {
        "ident": ident, "details": details, "condition": condition,
        "active": active, "stats": stats, "moves": moves,
        "baseAbility": ability, "ability": ability, "item": item,
        "pokeball": "pokeball",
    }
    if tera:
        d["teraType"] = tera
    return d


# our own side, every mon nicknamed something unrelated to its species
REQUEST = {
    "active": [{"moves": [
        {"move": "Flamethrower", "id": "flamethrower", "pp": 24, "maxpp": 24,
         "target": "normal", "disabled": False},
    ]}],
    "side": {"name": "wizbot", "id": "p1", "pokemon": [
        _req_mon("p1: Sparky", "Ninetales, L100, F", "323/323", True,
                 {"atk": 152, "def": 186, "spa": 240, "spd": 236, "spe": 298},
                 ["flamethrower"], "drought", "heatrock", tera="Grass"),
        _req_mon("p1: Bigfoot", "Slowking-Galar, L100, M", "394/394", False,
                 {"atk": 100, "def": 200, "spa": 250, "spd": 300, "spe": 90},
                 ["futuresight"], "regenerator", "heavydutyboots",
                 tera="Water"),
    ]},
}


def make_battle():
    b = Battle("battle-gen9ou-nick-1", "wizbot", logging.getLogger("test"),
               gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Sparky", "Ninetales, L100, F",
                     "323/323"])
    b.parse_message(["", "switch", "p2a: Steve", "Kingambit, L100, M",
                     "100/100"])
    b.parse_message(["", "turn", "1"])
    return b


def test_both_sides_translate_by_species_not_nickname():
    state = Gen9Translator().translate(make_battle())
    ours = {p.id for p in state.side_one.pokemon if p.id != "none"}
    theirs = {p.id for p in state.side_two.pokemon if p.id != "none"}
    assert "ninetales" in ours and "slowkinggalar" in ours
    assert "sparky" not in ours and "bigfoot" not in ours
    assert "kingambit" in theirs and "steve" not in theirs
    active = state.side_one.pokemon[int(state.side_one.active_index)]
    assert active.id == "ninetales"
    assert active.ability == "drought"      # real mon's data, not a stub


def test_own_tera_survives_a_nicknamed_request():
    # the request's teraType is mined by hand (poke-env drops it); keying
    # that map by the ident's NICKNAME while every lookup uses the species
    # silently loses our own tera types on any nicknamed team
    state = Gen9Translator().translate(make_battle())
    nine = next(p for p in state.side_one.pokemon if p.id == "ninetales")
    king = next(p for p in state.side_two.pokemon if p.id == "kingambit")
    assert nine.tera_type == "grass"        # from the request, not types[0]
    assert king.id == "kingambit"           # opponent unaffected


def test_inference_events_attribute_to_the_details_species():
    # a nicknamed forme must resolve to its DETAILS species; resolving by
    # ident would give "steve" (or, for the 08-04 forme bug, "slowking")
    # and every proof keyed to it would silently miss
    o = _obs(["|switch|p2a: Steve|Slowking-Galar, M|394/394",
              "|move|p2a: Steve|Chilly Reception|p1a: Sparky",
              "|turn|2"])
    assert o._event_species(
        ["", "-damage", "p2a: Steve", "345/394"]) == "slowkinggalar"
    # and a real proof lands on the species, not the nickname
    assert "assaultvest" in o.forbidden("slowkinggalar")
    assert "assaultvest" not in o.forbidden("steve")


def test_nicknamed_opponent_reveals_key_by_species():
    # a [from]-tagged reveal resolves its owner through the active table,
    # so the item lands on the species even when the ident is a nickname
    o = _obs(["|switch|p2a: Steve|Gliscor, M|100/100",
              "|-damage|p2a: Steve|94/100|[from] item: Life Orb",
              "|turn|3"])
    assert o.revealed_item.get("gliscor") == "lifeorb"
    assert "steve" not in o.revealed_item
