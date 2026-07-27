# Regression tests for the immune-attack no-op filter (gen9_player).
#
# A flat-eval MCTS can't tell a 0-damage attack from a real one, so without a
# filter the engine clicks immune moves for turns on end (observed live: a
# Gholdengo firing Shadow Ball into Blissey six turns running; two Gliscor in
# an Earthquake-off). _is_noop_attack drops those, with guards so it never
# drops a move that would actually land.

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.battle.battle import Battle

from showdown.gen9_player import (
    _is_noop_attack, _is_noop_prankster, _is_noop_ability, _is_noop_item)


def _battle(our, our_details, moves, our_ability, our_item, opp, opp_details):
    """A turn-1 battle: our active with `moves` (list of (name, id)) facing opp."""
    req = {"active": [{"moves": [
        {"move": m, "id": i, "pp": 16, "maxpp": 16, "target": "normal",
         "disabled": False} for m, i in moves]}],
        "side": {"name": "wizbot", "id": "p1", "pokemon": [{
            "ident": f"p1: {our}", "details": our_details, "condition": "350/350",
            "active": True, "moves": [i for _, i in moves],
            "stats": {"atk": 250, "def": 200, "spa": 250, "spd": 200, "spe": 200},
            "baseAbility": our_ability, "ability": our_ability, "item": our_item,
            "pokeball": "pokeball"}]}}
    b = Battle(f"b-{our}-{opp}", "wizbot", logging.getLogger("t"), gen=9)
    b.parse_request(req)
    b.parse_message(["", "switch", f"p1a: {our}", our_details, "350/350"])
    b.parse_message(["", "switch", f"p2a: {opp}", opp_details, "100/100"])
    b.parse_message(["", "turn", "1"])
    return b


def _move(b, mid):
    return next(m for m in b.available_moves if m.id == mid)


# ---- type immunity (certain, no prediction) ----

def test_ghost_into_normal_is_noop():
    b = _battle("Gholdengo", "Gholdengo, L100",
                [("Shadow Ball", "shadowball"), ("Make It Rain", "makeitrain")],
                "goodasgold", "airballoon", "Blissey", "Blissey, L100, F")
    assert _is_noop_attack(_move(b, "shadowball"), b) is True
    assert _is_noop_attack(_move(b, "makeitrain"), b) is False  # Steel hits


def test_ground_into_flying_type_is_noop():
    b = _battle("Gliscor", "Gliscor, L100, M", [("Earthquake", "earthquake")],
                "poisonheal", "toxicorb", "Gliscor", "Gliscor, L100, M")
    assert _is_noop_attack(_move(b, "earthquake"), b) is True


# ---- revealed ability immunity (only when revealed -> certain) ----

def test_levitate_only_when_revealed():
    b = _battle("Garchomp", "Garchomp, L100, M", [("Earthquake", "earthquake")],
                "roughskin", "lifeorb", "Bronzong", "Bronzong, L100")
    # unrevealed ability: chart says 2x, we do NOT presume Levitate
    assert _is_noop_attack(_move(b, "earthquake"), b) is False
    b.opponent_active_pokemon._ability = "levitate"
    assert _is_noop_attack(_move(b, "earthquake"), b) is True


def test_flash_fire_and_water_absorb():
    b = _battle("Volcarona", "Volcarona, L100, M", [("Flamethrower", "flamethrower")],
                "flamebody", "heavydutyboots", "Heatran", "Heatran, L100, M")
    b.opponent_active_pokemon._ability = "flashfire"
    assert _is_noop_attack(_move(b, "flamethrower"), b) is True

    b = _battle("Pelipper", "Pelipper, L100, M", [("Surf", "surf")],
                "drizzle", "damprock", "Clodsire", "Clodsire, L100, M")
    b.opponent_active_pokemon._ability = "waterabsorb"
    assert _is_noop_attack(_move(b, "surf"), b) is True


def test_air_balloon_vs_ground():
    b = _battle("Garchomp", "Garchomp, L100, M", [("Earthquake", "earthquake")],
                "roughskin", "lifeorb", "Heatran", "Heatran, L100, M")
    b.opponent_active_pokemon.item = "airballoon"
    assert _is_noop_attack(_move(b, "earthquake"), b) is True


# ---- guards: the immunity must not be lifted ----

def test_mold_breaker_ignores_levitate():
    b = _battle("Excadrill", "Excadrill, L100, M", [("Earthquake", "earthquake")],
                "moldbreaker", "lifeorb", "Bronzong", "Bronzong, L100")
    b.opponent_active_pokemon._ability = "levitate"
    assert _is_noop_attack(_move(b, "earthquake"), b) is False


def test_scrappy_hits_ghost():
    b = _battle("Kangaskhan", "Kangaskhan, L100, F", [("Body Slam", "bodyslam")],
                "scrappy", "silkscarf", "Gholdengo", "Gholdengo, L100")
    assert _is_noop_attack(_move(b, "bodyslam"), b) is False


def test_gravity_grounds_levitate():
    b = _battle("Garchomp", "Garchomp, L100, M", [("Earthquake", "earthquake")],
                "roughskin", "lifeorb", "Bronzong", "Bronzong, L100")
    b.opponent_active_pokemon._ability = "levitate"
    b.parse_message(["", "-fieldstart", "move: Gravity"])
    assert _is_noop_attack(_move(b, "earthquake"), b) is False


# ---- fixed-damage attacks respect immunity; self-buffs don't ----

def test_fixed_damage_moves_respect_immunity():
    # Seismic Toss (Fighting) whiffs on Ghost but HITS a Normal target
    b = _battle("Chansey", "Chansey, L100, F",
                [("Seismic Toss", "seismictoss")], "naturalcure", "eviolite",
                "Gholdengo", "Gholdengo, L100")
    assert _is_noop_attack(_move(b, "seismictoss"), b) is True
    b = _battle("Chansey", "Chansey, L100, F",
                [("Seismic Toss", "seismictoss")], "naturalcure", "eviolite",
                "Blissey", "Blissey, L100, F")
    assert _is_noop_attack(_move(b, "seismictoss"), b) is False
    # Night Shade (Ghost) whiffs on Normal
    b = _battle("Dragapult", "Dragapult, L100, M",
                [("Night Shade", "nightshade")], "clearbody", "lifeorb",
                "Blissey", "Blissey, L100, F")
    assert _is_noop_attack(_move(b, "nightshade"), b) is True


# ---- never flag Status self-buffs / switches ----

def test_status_move_and_switch_untouched():
    # Swords Dance is Normal-typed but a Status self-buff — fine into a Ghost
    b = _battle("Gholdengo", "Gholdengo, L100",
                [("Nasty Plot", "nastyplot"), ("Swords Dance", "swordsdance"),
                 ("Shadow Ball", "shadowball")],
                "goodasgold", "leftovers", "Gholdengo", "Gholdengo, L100")
    assert _is_noop_attack(_move(b, "nastyplot"), b) is False
    assert _is_noop_attack(_move(b, "swordsdance"), b) is False
    assert _is_noop_attack(None, b) is False                   # switch sentinel


# ---- Prankster status vs Dark (gen7+ immunity) ----

_PRANK_MOVES = [("Thunder Wave", "thunderwave"), ("Taunt", "taunt"),
                ("Spikes", "spikes"), ("Bulk Up", "bulkup")]


def test_prankster_status_blocked_by_dark():
    # Grimmsnarl (Prankster) into Dark/Steel Kingambit: foe-targeting status fails
    b = _battle("Grimmsnarl", "Grimmsnarl, L100, M", _PRANK_MOVES,
                "prankster", "leftovers", "Kingambit", "Kingambit, L100, M")
    assert _is_noop_prankster(_move(b, "thunderwave"), b) is True
    assert _is_noop_prankster(_move(b, "taunt"), b) is True
    # side hazards and self-buffs don't target the Dark mon -> not blocked
    assert _is_noop_prankster(_move(b, "spikes"), b) is False
    assert _is_noop_prankster(_move(b, "bulkup"), b) is False


def test_prankster_needs_prankster_and_dark():
    # non-Dark target: Thunder Wave lands
    b = _battle("Grimmsnarl", "Grimmsnarl, L100, M", _PRANK_MOVES,
                "prankster", "leftovers", "Dragonite", "Dragonite, L100, M")
    assert _is_noop_prankster(_move(b, "thunderwave"), b) is False
    # non-Prankster user: Dark blocks nothing
    b = _battle("Klefki", "Klefki, L100", _PRANK_MOVES,
                "frisk", "leftovers", "Kingambit", "Kingambit, L100, M")
    assert _is_noop_prankster(_move(b, "thunderwave"), b) is False


# ---- opponent ability nullifies / bounces the move ----

_AB_MOVES = [("Toxic", "toxic"), ("Spikes", "spikes"),
             ("Stealth Rock", "stealthrock"), ("Shadow Ball", "shadowball"),
             ("Boomburst", "boomburst"), ("Spore", "spore"),
             ("Bulk Up", "bulkup"), ("Earthquake", "earthquake")]


def test_good_as_gold_blocks_foe_status_only():
    # Gholdengo is single-ability, so Good as Gold is known on sight (certain)
    b = _battle("Clefable", "Clefable, L100, F", _AB_MOVES,
                "magicguard", "lifeorb", "Gholdengo", "Gholdengo, L100")
    assert _is_noop_ability(_move(b, "toxic"), b) is True
    assert _is_noop_ability(_move(b, "spikes"), b) is False       # side, not at it
    assert _is_noop_ability(_move(b, "shadowball"), b) is False   # damaging, not status


def test_magic_bounce_reflects_status_and_hazards():
    b = _battle("Ferrothorn", "Ferrothorn, L100, F", _AB_MOVES,
                "ironbarbs", "leftovers", "Hatterene", "Hatterene, L100, F")
    b.opponent_active_pokemon._ability = "magicbounce"
    assert _is_noop_ability(_move(b, "toxic"), b) is True
    assert _is_noop_ability(_move(b, "stealthrock"), b) is True   # hazard bounced
    assert _is_noop_ability(_move(b, "bulkup"), b) is False       # self


def test_flag_abilities_and_mold_breaker():
    b = _battle("Chien-Pao", "Chien-Pao, L100", _AB_MOVES,
                "swordofruin", "lifeorb", "Bellibolt", "Bellibolt, L100, M")
    b.opponent_active_pokemon._ability = "soundproof"
    assert _is_noop_ability(_move(b, "boomburst"), b) is True
    assert _is_noop_ability(_move(b, "earthquake"), b) is False
    # Bulletproof
    b = _battle("Dragapult", "Dragapult, L100, M", _AB_MOVES,
                "clearbody", "lifeorb", "Chesnaught", "Chesnaught, L100, M")
    b.opponent_active_pokemon._ability = "bulletproof"
    assert _is_noop_ability(_move(b, "shadowball"), b) is True
    # Mold Breaker lifts ability immunity wholesale
    b = _battle("Excadrill", "Excadrill, L100, M", _AB_MOVES,
                "moldbreaker", "lifeorb", "Gholdengo", "Gholdengo, L100")
    assert _is_noop_ability(_move(b, "toxic"), b) is False


def test_safety_goggles_item_blocks_powder():
    b = _battle("Amoonguss", "Amoonguss, L100, M", _AB_MOVES,
                "regenerator", "lifeorb", "Garchomp", "Garchomp, L100, M")
    b.opponent_active_pokemon.item = "safetygoggles"
    assert _is_noop_item(_move(b, "spore"), b) is True
    assert _is_noop_item(_move(b, "toxic"), b) is False           # not a powder move
    # items are NOT bypassed by Mold Breaker (unlike Overcoat, the ability twin)
    b = _battle("Excadrill", "Excadrill, L100, M", _AB_MOVES,
                "moldbreaker", "lifeorb", "Garchomp", "Garchomp, L100, M")
    b.opponent_active_pokemon.item = "safetygoggles"
    assert _is_noop_item(_move(b, "spore"), b) is True
