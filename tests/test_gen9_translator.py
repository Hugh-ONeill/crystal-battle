# Tests for the gen9 monotype translator (poke-env Battle -> poke-engine State).
#
# Battles are constructed by feeding real Showdown protocol shapes through
# poke-env's parser -- the same path a live connection uses -- so these tests
# also pin our assumptions about poke-env's message handling.

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator, parse_engine_choice


def _req_mon(ident, details, condition, active, stats, moves, ability, item):
    return {
        "ident": ident, "details": details, "condition": condition,
        "active": active, "stats": stats, "moves": moves,
        "baseAbility": ability, "ability": ability, "item": item,
        "pokeball": "pokeball",
    }


# A meta-realistic sun-Fire core (own side: full information via request).
REQUEST = {
    "active": [{"moves": [
        {"move": "Flamethrower", "id": "flamethrower", "pp": 24, "maxpp": 24,
         "target": "normal", "disabled": False},
        {"move": "Solar Beam", "id": "solarbeam", "pp": 16, "maxpp": 16,
         "target": "normal", "disabled": False},
        {"move": "Encore", "id": "encore", "pp": 8, "maxpp": 8,
         "target": "normal", "disabled": False},
        {"move": "Will-O-Wisp", "id": "willowisp", "pp": 24, "maxpp": 24,
         "target": "normal", "disabled": False},
    ]}],
    "side": {"name": "wizbot", "id": "p1", "pokemon": [
        _req_mon("p1: Ninetales", "Ninetales, L100, F", "323/323", True,
                 {"atk": 152, "def": 186, "spa": 240, "spd": 236, "spe": 298},
                 ["flamethrower", "solarbeam", "encore", "willowisp"],
                 "drought", "heatrock"),
        _req_mon("p1: Heatran", "Heatran, L100, M", "386/386", False,
                 {"atk": 194, "def": 248, "spa": 296, "spd": 248, "spe": 253},
                 ["magmastorm", "earthpower", "taunt", "stealthrock"],
                 "flashfire", "airballoon"),
        _req_mon("p1: Volcarona", "Volcarona, L100, M", "391/391", False,
                 {"atk": 140, "def": 166, "spa": 309, "spd": 246, "spe": 299},
                 ["quiverdance", "fierydance", "gigadrain", "morningsun"],
                 "flamebody", "heavydutyboots"),
    ]},
}


def make_battle():
    b = Battle("battle-gen9monotype-test-1", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F", "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    b.parse_message(["", "turn", "1"])
    return b


def _find(side, species):
    for p in side.pokemon:
        if p.id == species:
            return p
    raise AssertionError(f"{species} not found in side")


def test_own_side_exact():
    state = Gen9Translator().translate(make_battle())
    nine = state.side_one.pokemon[0]
    assert nine.id == "ninetales"          # active is slot 0
    assert nine.hp == 323 and nine.maxhp == 323
    assert nine.special_attack == 240 and nine.speed == 298
    assert nine.ability == "drought"
    assert nine.item == "heatrock"
    assert {m.id for m in nine.moves} == {"flamethrower", "solarbeam",
                                          "encore", "willowisp"}
    assert nine.types == ("fire", "typeless")
    assert not nine.terastallized
    heatran = _find(state.side_one, "heatran")
    assert heatran.item == "airballoon" and heatran.attack == 194


def test_side_conditions_boosts_weather():
    b = make_battle()
    b.parse_message(["", "-weather", "SunnyDay"])
    b.parse_message(["", "-sidestart", "p1: wizbot", "Spikes"])
    b.parse_message(["", "-sidestart", "p1: wizbot", "Spikes"])
    b.parse_message(["", "-sidestart", "p2: opponent", "move: Stealth Rock"])
    b.parse_message(["", "-boost", "p2a: Garchomp", "atk", "2"])
    state = Gen9Translator().translate(b)

    assert state.side_one.side_conditions.spikes == 2
    assert state.side_two.side_conditions.stealth_rock == 1
    assert state.side_two.attack_boost == 2
    assert state.side_one.attack_boost == 0
    assert state.weather == "sun"
    # Ninetales' revealed Heat Rock -> 8-turn sun (set this turn: all 8 left)
    assert state.weather_turns_remaining == 8


def test_opp_canonical_fill():
    b = make_battle()
    # a second dragon pins the opponent's monotype
    b.parse_message(["", "switch", "p2a: Latios", "Latios, L100", "100/100"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    state = Gen9Translator().translate(b)

    chomp = _find(state.side_two, "garchomp")
    move_ids = [m.id for m in chomp.moves]
    assert "earthquake" in move_ids            # revealed move kept
    assert "none" not in move_ids              # canonical set filled the rest
    assert chomp.ability != "noability"        # canonical ability guessed
    # canonical spread, not the neutral-85 fallback: Garchomp neutral-85 HP
    # is 379; any real spread differs
    assert 300 < chomp.maxhp < 450
    assert chomp.hp == chomp.maxhp             # 100/100 fraction preserved


def test_status_substitute_and_volatiles():
    b = make_battle()
    b.parse_message(["", "-status", "p2a: Garchomp", "brn"])
    b.parse_message(["", "-start", "p2a: Garchomp", "Substitute"])
    b.parse_message(["", "-start", "p1a: Ninetales", "move: Taunt"])
    state = Gen9Translator().translate(b)

    chomp = state.side_two.pokemon[0]
    assert chomp.status == "brn"
    assert "substitute" in state.side_two.volatile_statuses
    assert state.side_two.substitute_health == chomp.maxhp // 4
    assert "taunt" in state.side_one.volatile_statuses
    assert state.side_one.volatile_status_durations.taunt >= 1


def test_unrevealed_slots_are_fainted_dummies():
    # monotype source: no team prediction yet -> dummies (bench-validated)
    state = Gen9Translator().translate(make_battle())
    assert len(state.side_two.pokemon) == 6
    for filler in state.side_two.pokemon[1:]:
        assert filler.hp == 0


def test_gen9ou_unrevealed_slots_are_predicted():
    # chaos source: unrevealed slots get teammate-correlated predictions at
    # full HP -- fainted dummies made the engine's eval treat them as dead
    # and play with unearned aggression (0-10 vs foul-play)
    state = Gen9Translator(set_source="gen9ou").translate(make_battle())
    assert len(state.side_two.pokemon) == 6
    alive = [p for p in state.side_two.pokemon if p.hp > 0]
    assert len(alive) == 6
    for predicted in state.side_two.pokemon[1:]:
        assert predicted.id
        assert any(m.id != "none" for m in predicted.moves)
    # searchable end-to-end
    assert pe.monte_carlo_tree_search(state, 20).side_one


def test_state_is_searchable():
    b = make_battle()
    b.parse_message(["", "-weather", "SunnyDay"])
    b.parse_message(["", "-sidestart", "p2: opponent", "move: Stealth Rock"])
    state = Gen9Translator().translate(b)

    # round-trips through the engine's serialization
    rebuilt = pe.State.from_string(state.to_string())
    assert rebuilt.side_one.pokemon[0].id.lower() == "ninetales"

    # and MCTS accepts it end-to-end
    result = pe.monte_carlo_tree_search(state, 50)
    assert result.side_one
    choices = {r.move_choice for r in result.side_one}
    assert any(not c.endswith("-tera") for c in choices)


def test_encore_sets_last_used_move():
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Latios", "Latios, L100", "100/100"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    b.parse_message(["", "-start", "p2a: Garchomp", "Encore"])
    state = Gen9Translator().translate(b)

    # engine panics if ENCORE is set without last_used_move pointing at a slot
    assert "encore" in state.side_two.volatile_statuses
    slot = int(state.side_two.last_used_move.split(":")[1])
    assert state.side_two.pokemon[0].moves[slot].id == "earthquake"
    assert pe.monte_carlo_tree_search(state, 20).side_two


def test_encore_without_known_move_is_dropped():
    b = make_battle()
    b.parse_message(["", "-start", "p2a: Garchomp", "Encore"])
    state = Gen9Translator().translate(b)
    assert "encore" not in state.side_two.volatile_statuses
    assert pe.monte_carlo_tree_search(state, 20).side_one


def test_gen9ou_chaos_set_fill():
    b = make_battle()
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = _find(state.side_two, "garchomp")
    move_ids = [m.id for m in chomp.moves]
    assert "earthquake" in move_ids            # revealed move kept
    assert "none" not in move_ids              # chaos stats filled the rest
    assert chomp.ability != "noability"
    # un-tera'd opponent carries the chaos-predicted tera type
    assert not chomp.terastallized
    assert chomp.tera_type not in ("", "typeless")


def test_opponent_terastallize_reflected():
    b = make_battle()
    b.parse_message(["", "-terastallize", "p2a: Garchomp", "Fire"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = state.side_two.pokemon[0]
    assert chomp.terastallized
    assert chomp.tera_type == "fire"
    # base types stay base (the engine applies tera itself)
    assert chomp.types == ("dragon", "ground")


def test_own_tera_type_mined_from_request():
    # REQUEST carries no teraType key -> fallback is first base type;
    # with teraType present it should be used
    req = {**REQUEST, "side": {**REQUEST["side"], "pokemon": [
        {**REQUEST["side"]["pokemon"][0], "teraType": "Grass"},
        *REQUEST["side"]["pokemon"][1:],
    ]}}
    b = Battle("battle-gen9monotype-test-2", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(req)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F", "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    state = Gen9Translator(set_source="gen9ou").translate(b)
    nine = state.side_one.pokemon[0]
    assert not nine.terastallized
    assert nine.tera_type == "grass"


def test_opponent_choice_lock_disables_other_moves():
    b = make_battle()
    b.parse_message(["", "-item", "p2a: Garchomp", "Choice Scarf"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = state.side_two.pokemon[0]
    assert chomp.item == "choicescarf"
    states = {m.id: m.disabled for m in chomp.moves if m.id != "none"}
    assert states["earthquake"] is False
    assert all(disabled for mid, disabled in states.items() if mid != "earthquake")
    assert state.side_two.last_used_move == "move:0"


def test_no_lock_without_choice_item():
    b = make_battle()
    b.parse_message(["", "-item", "p2a: Garchomp", "Leftovers"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = state.side_two.pokemon[0]
    assert not any(m.disabled for m in chomp.moves)
    assert state.side_two.last_used_move == "move:0"  # still fed to the engine


def test_switch_clears_choice_lock():
    b = make_battle()
    b.parse_message(["", "-item", "p2a: Garchomp", "Choice Scarf"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    b.parse_message(["", "switch", "p2a: Latios", "Latios, L100", "100/100"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = _find(state.side_two, "garchomp")
    assert not any(m.disabled for m in chomp.moves)
    assert not state.side_two.last_used_move.startswith("move:0")


def test_own_request_restrictions_carry_into_state():
    # a choice-locked request lists only the locked move as available
    b = make_battle()
    locked = {**REQUEST, "active": [{"moves": [
        {"move": "Flamethrower", "id": "flamethrower", "pp": 23, "maxpp": 24,
         "target": "normal", "disabled": False}]}]}
    b.parse_request(locked)
    state = Gen9Translator().translate(b)

    nine = state.side_one.pokemon[0]
    states = {m.id: m.disabled for m in nine.moves}
    assert states["flamethrower"] is False
    assert states["solarbeam"] and states["encore"] and states["willowisp"]


def test_preview_order_string():
    from showdown.gen9_player import _preview_order
    assert _preview_order(0, 6) == "/team 123456"
    assert _preview_order(2, 6) == "/team 312456"
    assert _preview_order(5, 6) == "/team 612345"


def test_predicted_preview_paste_parses_and_searches():
    from showdown.local_battle import parse_showdown_team, build_pe_state_gen9
    from monotype.lead_picker import pick_leads

    tr = Gen9Translator(set_source="gen9ou")
    paste = tr.predicted_preview_paste(
        ["Gholdengo", "Great Tusk", "Kingambit", "Dragapult", "Zamazenta",
         "Iron Valiant"])
    mons = parse_showdown_team(paste)
    assert len(mons) == 6
    for m in mons:
        assert len(m["moves"]) == 4
        assert m["ability"]
    # end-to-end: usable by the lead maximin at preview time
    ours = (Path(__file__).parent.parent / "showdown" / "teams"
            / "gen9ou_sample.txt").read_text()
    lead, _, matrix = pick_leads(ours, paste, search_ms=5)
    assert 0 <= lead < 6
    assert len(matrix) == 6 and len(matrix[0]) == 6


def test_chaos_sample_set_varies_and_respects_known_moves():
    import random
    from showdown.chaos_stats import ChaosStats
    stats = ChaosStats(format="gen9ou").pokemon["garchomp"]

    rng = random.Random(7)
    items = {stats.sample_set(rng)["item"] for _ in range(40)}
    assert len(items) >= 2  # actually sampling, not top-1

    s = stats.sample_set(random.Random(1), known_moves=("earthquake",))
    assert "earthquake" not in s["moves"]
    assert len(s["moves"]) <= 3  # revealed move occupies a slot
    assert s["ability"] and s["item"]


def test_sampled_translation_varies_and_searches():
    import random
    b = make_battle()
    tr = Gen9Translator(set_source="gen9ou")

    benches = set()
    for seed in range(6):
        state = tr.translate(b, rng=random.Random(seed))
        benches.add(tuple(p.id for p in state.side_two.pokemon))
        assert all(p.hp > 0 for p in state.side_two.pokemon)
    assert len(benches) >= 2  # different sampled worlds
    assert pe.monte_carlo_tree_search(state, 20).side_one

    # rng=None stays deterministic
    a = tr.translate(b, rng=None)
    c = tr.translate(b, rng=None)
    assert [p.id for p in a.side_two.pokemon] == [p.id for p in c.side_two.pokemon]


def test_merge_mcts_results():
    from types import SimpleNamespace as NS
    from showdown.gen9_player import _merge_mcts_results
    r1 = NS(side_one=[NS(move_choice="earthquake", visits=100, total_score=60.0),
                      NS(move_choice="switch heatran", visits=50, total_score=20.0)])
    r2 = NS(side_one=[NS(move_choice="earthquake", visits=80, total_score=30.0),
                      NS(move_choice="protect", visits=120, total_score=70.0)])
    merged = _merge_mcts_results([r1, r2])
    by_choice = {m.move_choice: m for m in merged}
    assert by_choice["earthquake"].visits == 180
    assert merged[0].move_choice == "earthquake"  # 180 > 120 > 50
    assert by_choice["protect"].visits == 120


def test_speed_floor_infers_scarf():
    # Amoonguss (base 30 spe, modeled far slower than Ninetales' 298) moves
    # FIRST at equal priority -> floor contradiction -> inferred scarf
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Amoonguss", "Amoonguss, F", "100/100"])
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p2a: Amoonguss", "Sludge Bomb", "p1a: Ninetales"])
    b.parse_message(["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Amoonguss"])
    b.parse_message(["", "turn", "2"])
    state = Gen9Translator(set_source="gen9ou").translate(b)
    fungus = _find(state.side_two, "amoonguss")
    assert fungus.item == "choicescarf"


def test_scarf_inference_records_confirmed_belief():
    # the real translator must stamp obs.confirmed when it adopts the
    # inferred scarf — this is the link the live player diffs into a
    # "that's a Scarf" commentary beat (belief-delta pipeline)
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Amoonguss", "Amoonguss, F", "100/100"])
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p2a: Amoonguss", "Sludge Bomb", "p1a: Ninetales"])
    b.parse_message(["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Amoonguss"])
    b.parse_message(["", "turn", "2"])
    tr = Gen9Translator(set_source="gen9ou")
    tr.translate(b)
    assert tr._obs.confirmed.get("amoonguss") == "choicescarf"

    # and the player's diff turns that into exactly one set_reveal beat
    from showdown.gen9_player import Gen9PokeEnginePlayer as P
    from showdown.beat_director import Director, classify
    p = P.__new__(P)
    p._airi = object()
    p._director = Director()
    p._announced_beliefs = {}
    p._translator = tr
    p._emit_belief_deltas()
    reveals = [c for c in (classify(ev) for ev in p._director._pending)
               if c and c.beat == "set_reveal"]
    assert len(reveals) == 1 and "Choice Scarf" in reveals[0].prose


def test_boots_inference_records_confirmed_belief():
    # opponent switches Corviknight in over our Stealth Rock and takes no
    # chip -> the real translator adopts Heavy-Duty Boots and stamps it
    b = make_battle()
    b.parse_message(["", "move", "p1a: Ninetales", "Stealth Rock"])
    b.parse_message(["", "-sidestart", "p2: opp", "move: Stealth Rock"])
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "switch", "p2a: Corviknight", "Corviknight, M", "100/100"])
    b.parse_message(["", "turn", "2"])
    tr = Gen9Translator(set_source="gen9ou")
    tr.translate(b)
    assert tr._obs.boots_inferred("corviknight") == "heavydutyboots"
    assert tr._obs.confirmed.get("corviknight") == "heavydutyboots"
    corv = _find(tr.translate(b).side_two, "corviknight")
    assert corv.item == "heavydutyboots"


def test_no_scarf_when_we_moved_first():
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Amoonguss", "Amoonguss, F", "100/100"])
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Amoonguss"])
    b.parse_message(["", "move", "p2a: Amoonguss", "Sludge Bomb", "p1a: Ninetales"])
    b.parse_message(["", "turn", "2"])
    state = Gen9Translator(set_source="gen9ou").translate(b)
    fungus = _find(state.side_two, "amoonguss")
    assert fungus.item != "choicescarf"


def test_speed_ceiling_clamps_modeled_speed():
    # Dragapult (modeled ~1.5x faster than Ninetales) moves AFTER us at
    # equal priority -> ceiling -> stat clamped below our speed (covers
    # slow spreads and Iron Ball-style items without naming them)
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Dragapult", "Dragapult, F", "100/100"])
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Dragapult"])
    b.parse_message(["", "move", "p2a: Dragapult", "Dragon Darts", "p1a: Ninetales"])
    b.parse_message(["", "turn", "2"])
    state = Gen9Translator(set_source="gen9ou").translate(b)
    pult = _find(state.side_two, "dragapult")
    assert pult.speed < 298  # our Ninetales' speed from the request


def test_damage_bracket_upgrades_item():
    # a weak move (U-turn) hitting for 1.6x the modeled max roll fits within
    # our HP (overkill damage is HP-censored, so strong moves can't prove a
    # boost) -> Choice Band inferred from the protocol alone
    from showdown.set_inference import BattleObservations

    probe_b = make_battle()
    probe_b.parse_message(["", "switch", "p2a: Pelipper", "Pelipper, M", "100/100"])
    tr = Gen9Translator(set_source="gen9ou")
    probe_state = tr.translate(probe_b)
    bird = _find(probe_state.side_two, "pelipper")
    ev = {"species": "pelipper", "move": "uturn", "our_species": "ninetales",
          "weather": "none", "se": False, "damage": 100}
    ratio = BattleObservations()._observed_ratio(ev, bird, tr._my_built)
    max_roll = 100 / ratio
    hit = int(max_roll * 1.6)
    assert hit < 323  # must not be HP-censored

    b = make_battle()
    b.parse_message(["", "switch", "p2a: Pelipper", "Pelipper, M", "100/100"])
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p2a: Pelipper", "U-turn", "p1a: Ninetales"])
    b.parse_message(["", "-damage", "p1a: Ninetales", f"{323 - hit}/323"])
    b.parse_message(["", "turn", "2"])
    state = Gen9Translator(set_source="gen9ou").translate(b)
    bird = _find(state.side_two, "pelipper")
    assert bird.item == "choiceband"


def test_damage_bracket_tiers():
    # decision-level check with controlled ratios: moderate -> lifeorb,
    # big -> choice; SE-only boost with clean non-SE -> expert belt
    import copy
    from showdown.set_inference import BattleObservations
    tr = Gen9Translator(set_source="gen9ou")
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Pelipper", "Pelipper, M", "100/100"])
    state = tr.translate(b)  # populates _my_built
    bird = _find(state.side_two, "pelipper")
    our = tr._my_built

    obs = BattleObservations()
    base = {"species": "pelipper", "move": "hydropump",
            "our_species": "ninetales", "weather": "none", "se": True}
    probe_ratio = obs._observed_ratio({**base, "damage": 100}, bird, our)
    assert probe_ratio is not None and probe_ratio > 0  # damage-calc sides OK
    max_roll = 100 / probe_ratio

    obs.damage_evidence = [{**base, "damage": int(max_roll * 1.25)}]
    assert obs.damage_item_upgrade("pelipper", bird, our) == "lifeorb"

    obs.damage_evidence = [{**base, "damage": int(max_roll * 1.6)}]
    assert obs.damage_item_upgrade("pelipper", bird, our) == "choicespecs"

    clean_nonse = {**base, "move": "uturn", "se": False, "damage": 1}
    obs.damage_evidence = [{**base, "damage": int(max_roll * 1.25)}, clean_nonse]
    assert obs.damage_item_upgrade("pelipper", bird, our) == "expertbelt"


def test_speed_pessimistic_sampling():
    import random
    from showdown.chaos_stats import ChaosStats
    stats = ChaosStats(format="gen9ou").pokemon["ironvaliant"]
    s = stats.sample_set(random.Random(3), speed_pessimistic=True)
    assert s["item"] == "choicescarf"   # 2.3% usage clears the 2% bar
    assert s["evs"]["spe"] == 252       # fastest listed spread

    # and it flows through a pessimistic sampled translation
    b = make_battle()
    tr = Gen9Translator(set_source="gen9ou")
    state = tr.translate(b, rng=random.Random(1), speed_pessimistic=True)
    chomp = _find(state.side_two, "garchomp")
    assert chomp.item == "choicescarf"  # garchomp scarf 3.5% >= 2%


def test_force_switch_state_searches_replacements():
    b = make_battle()
    b.parse_message(["", "faint", "p1a: Ninetales"])
    fs_request = {
        "forceSwitch": [True],
        "side": {**REQUEST["side"], "pokemon": [
            {**REQUEST["side"]["pokemon"][0], "condition": "0 fnt"},
            *REQUEST["side"]["pokemon"][1:],
        ]},
    }
    b.parse_request(fs_request)
    assert b.force_switch

    state = Gen9Translator(set_source="gen9ou").translate(b)
    assert state.side_one.force_switch
    # the fainted active keeps its STABLE slot and active_index points at it;
    # the engine reads hp<=0 there and offers the replacements itself. (The
    # old convention packed a fainted dummy into slot 0 instead, which is
    # what made the array permute on every switch and broke tree reuse.)
    assert state.side_one.pokemon[int(state.side_one.active_index)].hp == 0
    alive = [p.id for p in state.side_one.pokemon if p.hp > 0]
    assert set(alive) == {"heatran", "volcarona"}

    result = pe.monte_carlo_tree_search(state, 50)
    top = sorted(result.side_one, key=lambda r: -r.visits)[:2]
    assert all(r.move_choice.startswith("switch ") for r in top)


def test_ps_sets_index_parses_and_filters():
    from showdown.ps_sets import get_index
    idx = get_index("gen9ou")
    cands = idx.consistent("gholdengo")
    assert len(cands) >= 4
    for c in cands:
        assert len(c["moves"]) == 4 and c["item"] != "" and c["evs"]

    # revealed move narrows to candidates carrying it
    narrowed = idx.consistent("gholdengo", known_moves=("thunderwave",))
    assert narrowed and all("thunderwave" in c["moves"] for c in narrowed)
    # revealed item pins the matching joint set
    balloon = idx.consistent("gholdengo", known_item="Air Balloon")
    assert balloon and all(c["item"] == "airballoon" for c in balloon)
    # an impossible speed floor eliminates everything
    assert idx.consistent("amoonguss", speed_floor=500) == []


def test_opp_set_prefers_joint_ps_sets():
    from showdown.ps_sets import get_index
    tr = Gen9Translator(set_source="gen9ou")
    got = tr._opp_set("gholdengo")
    assert got is not None
    ps_movesets = {tuple(c["moves"]) for c in
                   get_index("gen9ou").consistent("gholdengo")}
    # the returned set is one curated JOINT set, not composed marginals
    assert tuple(got["moves"]) in ps_movesets


def test_prefer_ps_false_uses_chaos_tier():
    from showdown.chaos_stats import ChaosStats
    tr = Gen9Translator(set_source="gen9ou")
    tr._prefer_ps = False
    got = tr._opp_set("gholdengo")
    tr._prefer_ps = True
    chaos_top = ChaosStats(format="gen9ou").pokemon["gholdengo"].top_moves(4)
    assert got["moves"] == chaos_top  # chaos marginals, not a curated set


def test_ps_pessimistic_picks_fastest_candidate():
    import random
    tr = Gen9Translator(set_source="gen9ou")
    tr._rng = random.Random(1)
    tr._speed_pess = True
    got = tr._opp_set("garchomp")
    tr._rng = None
    tr._speed_pess = False
    # garchomp scarf usage (3.5%) clears the pessimism bar
    assert got["item"] == "choicescarf"


def test_replay_sets_index():
    from showdown.replay_sets import get_index
    idx = get_index("gen9ou")
    assert idx is not None and idx.replays > 1000

    frags = idx.movesets("greattusk")
    assert frags and all(isinstance(f[0], tuple) and f[1] >= 1 for f in frags)
    # consistency: fragments must contain all revealed moves
    known = frags[0][0][:1]
    narrowed = idx.movesets("greattusk", known_moves=known)
    assert narrowed and all(set(known) <= set(f[0]) for f in narrowed)
    # deterministic pick is the most common consistent fragment
    picked = idx.pick_moves("greattusk", known_moves=known)
    assert picked and set(known) <= set(picked)

    # archetype matching round-trips a real key from the corpus
    key = next(iter(idx.teams))
    assert idx.team_match(key.split("|")) is not None
    assert idx.team_match(["pikachu"] * 6) is None


def test_chaos_tier_uses_replay_fragments():
    from showdown.replay_sets import get_index
    tr = Gen9Translator(set_source="gen9ou")
    tr._prefer_ps = False
    got = tr._opp_set("greattusk")
    tr._prefer_ps = True
    expected = get_index("gen9ou").pick_moves("greattusk")
    # observed fragment leads the moveset; chaos pads the remainder
    assert all(m in got["moves"] for m in expected[:4])


def test_data_tiers_off_reproduces_pure_chaos():
    from showdown.chaos_stats import ChaosStats
    tr = Gen9Translator(set_source="gen9ou", use_data_tiers=False)
    got = tr._opp_set("gholdengo")
    stats = ChaosStats(format="gen9ou").pokemon["gholdengo"]
    # exact pre-tier behavior: chaos top values, no curated set, no fragments
    assert got["moves"] == stats.top_moves(4)
    assert got["item"] == stats.top_item()


def test_confidence_gates():
    from showdown.replay_sets import get_index as replay_index
    from showdown.ps_sets import get_index as ps_index
    ridx = replay_index("gen9ou")
    pidx = ps_index("gen9ou")

    # corroboration is move-level (fragments are mostly partial): commonly
    # observed moves corroborate; a fabricated moveset does not
    assert ridx.corroborates("greattusk",
                             ["headlongrush", "rapidspin", "closecombat",
                              "icespinner"])
    assert not ridx.corroborates("greattusk",
                                 ["splash", "tackle", "growl", "pound"])

    # unanchored fragments need >= 3 sightings; anchored ones do not
    sparse = next((sp for sp, e in ridx.species.items()
                   if e["movesets"] and max(c for _, c in e["movesets"]) < 3),
                  None)
    if sparse is not None:
        assert ridx.pick_moves(sparse) is None  # gated
        anchor = ridx.species[sparse]["movesets"][0][0][0]
        assert ridx.pick_moves(sparse, known_moves=(anchor,)) is not None

    # PS tier without reveals: only corpus-corroborated candidates survive.
    # A species with PS sets but zero corpus presence gates to None.
    tr = Gen9Translator(set_source="gen9ou")
    ghost = next((sp for sp in pidx.candidates
                  if sp not in ridx.species), None)
    if ghost is not None:
        assert tr._ps_candidate(ghost, (), None, None) is None
    # with a revealed move, reveals themselves are the evidence
    got = tr._ps_candidate("gholdengo", ("shadowball",), None, None)
    assert got is not None and "shadowball" in got["moves"]

    # archetype gate: count-2 rosters are ignored, count>=3 accepted
    weak = next((k for k, v in ridx.teams.items() if v["count"] == 2), None)
    strong = next((k for k, v in ridx.teams.items() if v["count"] >= 3), None)
    class FakeBattle:
        teampreview_opponent_team = []
        opponent_team = {}
    if weak:
        b = FakeBattle()
        b.teampreview_opponent_team = [type("M", (), {"species": s})()
                                       for s in weak.split("|")]
        tr._resolve_archetype(b)
        assert tr._archetype is None
    if strong:
        b = FakeBattle()
        b.teampreview_opponent_team = [type("M", (), {"species": s})()
                                       for s in strong.split("|")]
        tr._resolve_archetype(b)
        assert tr._archetype is not None


def test_probabilistic_selection():
    import random
    from types import SimpleNamespace as NS
    from showdown.gen9_player import _select_choice, _lead_pool

    mappable = [(NS(visits=1000), "a"), (NS(visits=800), "b"),
                (NS(visits=740), "c"), (NS(visits=100), "d")]
    # 75% rule: pool is a+b (800 >= 750, 740 < 750); d never drawn
    rng = random.Random(5)
    seen = {_select_choice(mappable, rng)[1] for _ in range(200)}
    assert seen == {"a", "b"}
    # argmax mode is deterministic top
    assert _select_choice(mappable, rng, sample=False)[1] == "a"

    # lead pool: near-ties on worst-case row values within epsilon
    matrix = [[0.5, 0.2], [0.4, 0.18], [0.9, -0.5], [0.1, 0.0],
              [0.3, 0.15], [0.2, -0.1]]
    pool = _lead_pool(matrix, epsilon=0.08)
    assert 0 in pool and 1 in pool      # 0.2 and 0.18 within 0.08 of best
    assert 2 not in pool and 5 not in pool


def test_time_left_parser():
    from types import SimpleNamespace as NS
    from showdown.gen9_player import _time_left
    # NB: the wire protocol splits on '|', so the timer text (which itself
    # contains pipes) arrives SCATTERED across fields — the true poke-env
    # shape is ["", "inactive", "Time left: 300 sec this turn ", " 300 sec
    # total ", ...]. Feeding the unsplit string here once masked a live bug.
    b = NS(_replay_data=[
        ["", "inactive", "Battle timer is ON."],
        ["", "inactive", "Time left: 300 sec this turn ", " 300 sec total ",
         " 60 sec grace"],
        ["", "inactive", "CBGen9 has 270 seconds left."],
        ["", "inactive", "CBGen9 has 150 seconds left."],
    ])
    assert _time_left(b, "CBGen9") == 150       # most recent our-bank reading
    assert _time_left(NS(_replay_data=[]), "CBGen9") is None
    # falls back to 'sec total' when no per-player line matches our name
    b2 = NS(_replay_data=[
        ["", "inactive", "Time left: 200 sec this turn ", " 200 sec total"]])
    assert _time_left(b2, "CBGen9") == 200


def test_adaptive_escalation_decision():
    from types import SimpleNamespace as NS
    from showdown.gen9_player import Gen9PokeEnginePlayer as P
    import showdown.gen9_player as gp

    peaked = [NS(side_one=[NS(move_choice="a", visits=900, total_score=800),
                           NS(move_choice="b", visits=100, total_score=40)])]
    flat = [NS(side_one=[NS(move_choice="a", visits=520, total_score=300),
                         NS(move_choice="b", visits=480, total_score=270)])]

    calls = []
    stub = P.__new__(P)
    stub._search_ms, stub._escalate_ms = 300, 2000
    stub._base_frac, stub._base_max_ms = 0.02, 2000
    stub._grind_turn, stub._grind_max_ms = 20, 6000
    stub._collapse_turn, stub._collapse_moves = 25, 14
    stub._collapse_mons = 5
    stub._set_samples = 2
    stub._flat_threshold, stub._clock_floor_s = 0.55, 40
    stub._set_samples, stub._verbose = 2, False
    stub._escalate_bank_s, stub._bank_used_s = 90.0, 0.0
    stub._spend_frac, stub._escalate_max_ms = 0.25, 15000
    stub._escalate_min_turn, stub._escalate_min_gap = 20, 8
    stub._last_escalate_turn = -999
    stub._airi, stub._airi_last_sent = None, 0.0  # Airi bridge off in tests

    def fake_search(battle, ms=None, use_value=None):
        calls.append((ms, use_value))
        # probe runs use_value=False; escalation runs use_value=True
        return battle._probe if use_value is False else [NS(side_one=[])]
    stub._search_samples = fake_search

    def clock(bank, cap=None):
        gp._parse_clock = lambda *a: (bank, cap)

    # budget-by-clock: base probe budget scales with the parsed bank
    clock(1500)
    assert stub._base_budget_ms(NS(_replay_data=[])) == 2000   # capped
    clock(120)
    assert stub._base_budget_ms(NS(_replay_data=[])) == 1600   # (120-40)*2%
    clock(90)
    assert stub._base_budget_ms(NS(_replay_data=[])) == 1000
    clock(20)
    assert stub._base_budget_ms(NS(_replay_data=[])) == 150    # survival floor
    clock(None)
    assert stub._base_budget_ms(NS(_replay_data=[])) == 300    # no timer: pinned
    # grind-aware cap: from grind_turn on, cap rises to grind_max_ms —
    # but only when the bank affords it (2%-of-surplus rule still limits)
    clock(1500)
    assert stub._base_budget_ms(NS(_replay_data=[], turn=30)) == 6000
    clock(1500)
    assert stub._base_budget_ms(NS(_replay_data=[], turn=10)) == 2000
    clock(120)   # (120-40)*2% = 1.6s < both caps: unchanged late
    assert stub._base_budget_ms(NS(_replay_data=[], turn=30)) == 1600

    # decisive probe: no escalation regardless of clock
    calls.clear(); clock(120)
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=peaked))
    assert calls == [(1600, False)]

    # flat + server bank 120: surplus 80 * 0.25 = 20s -> capped at max 15000ms
    calls.clear(); clock(120)
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert (1600, False) in calls and (15000, True) in calls

    # flat + per-turn cap 10s bounds the spend: min(20, 10 - 1.6 - 5) = 3.4s
    calls.clear(); clock(120, cap=10)
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert (3400, True) in calls

    # flat + low bank: below the floor, survival-speed probe only
    calls.clear(); clock(20)
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert calls == [(150, False)]

    # no timer messages -> configured base + fallback fixed bank at escalate_ms
    calls.clear(); clock(None)
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert (300, False) in calls and (2000, True) in calls
    # ...and the fallback bank exhausts
    calls.clear(); stub._bank_used_s = 90.0
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert calls == [(300, False)]
    stub._bank_used_s = 0.0

    # opening flatness (turn < min_turn) is skipped
    calls.clear(); clock(120)
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=10, _replay_data=[], _probe=flat))
    assert calls == [(1600, False)]

    # min_gap spacing, then a legal late escalation updates the gap marker
    calls.clear()
    stub._last_escalate_turn = 100
    stub._adaptive_search(NS(turn=104, _replay_data=[], _probe=flat))
    assert calls == [(1600, False)]
    calls.clear()
    stub._adaptive_search(NS(turn=110, _replay_data=[], _probe=flat))
    assert (1600, False) in calls and (15000, True) in calls
    assert stub._last_escalate_turn == 110


def test_noop_hazard_filter():
    from types import SimpleNamespace as NS
    from poke_env.battle.side_condition import SideCondition
    from showdown.gen9_player import _is_noop_hazard

    # spikes at 3 (max) on opponent -> re-setting is a no-op
    b = NS(opponent_side_conditions={SideCondition.SPIKES: 3})
    assert _is_noop_hazard("spikes", b)
    # spikes at 2 -> still useful (a 3rd layer helps)
    b2 = NS(opponent_side_conditions={SideCondition.SPIKES: 2})
    assert not _is_noop_hazard("spikes", b2)
    # stealth rock present (max 1) -> no-op
    b3 = NS(opponent_side_conditions={SideCondition.STEALTH_ROCK: 1})
    assert _is_noop_hazard("stealthrock", b3)
    # non-hazard and switch (None) never filtered
    assert not _is_noop_hazard("earthquake", b3)
    assert not _is_noop_hazard(None, b3)


def test_noop_status_filter():
    from types import SimpleNamespace as NS
    from poke_env.battle.effect import Effect
    from poke_env.battle.pokemon_type import PokemonType
    from poke_env.battle.status import Status
    from showdown.gen9_player import _is_noop_status

    def opp(**kw):
        base = dict(effects={}, status=None, type_1=None, type_2=None,
                    ability=None)
        base.update(kw)
        return NS(opponent_active_pokemon=NS(**base))

    # toxic vs steel type -> immune -> no-op
    assert _is_noop_status("toxic", opp(type_1=PokemonType.STEEL))
    # toxic vs a normal-type -> lands -> not no-op
    assert not _is_noop_status("toxic", opp(type_1=PokemonType.NORMAL))
    # any status vs already-statused target -> no-op
    assert _is_noop_status("thunderwave",
                           opp(type_1=PokemonType.NORMAL, status=Status.BRN))
    # willowisp vs fire -> immune
    assert _is_noop_status("willowisp", opp(type_1=PokemonType.FIRE))
    # behind a substitute -> no-op
    assert _is_noop_status("toxic",
                           opp(type_1=PokemonType.NORMAL,
                               effects={Effect.SUBSTITUTE: 1}))
    # thunderwave vs ground -> immune; vs limber ability -> immune
    assert _is_noop_status("thunderwave", opp(type_1=PokemonType.GROUND))
    assert _is_noop_status("thunderwave",
                           opp(type_1=PokemonType.NORMAL, ability="limber"))
    # non-status move never filtered
    assert not _is_noop_status("earthquake", opp(type_1=PokemonType.STEEL))


def test_parse_engine_choice():
    assert parse_engine_choice("switch heatran") == ("switch", "heatran")
    assert parse_engine_choice("flamethrower") == ("move", "flamethrower")


def test_timer_variant_format_normalized():
    # PokeAgent runs gen9ou under timer-variant queues; these are mechanically
    # identical to the base tier and MUST resolve to its data files. A raw
    # "gen9oulongtimer" set_source sent the chaos fallback looking for a
    # nonexistent gen9oulongtimer_chaos.json -> FileNotFoundError -> random
    # moves mid-game (the maiden overnight run, turn 12 on).
    from showdown.gen9_translator import Gen9Translator, _base_format
    assert _base_format("gen9oulongtimer") == "gen9ou"
    assert _base_format("gen9oushorttimer") == "gen9ou"
    assert _base_format("gen1oulongtimer") == "gen1ou"
    assert _base_format("gen9ou") == "gen9ou"
    assert _base_format("monotype") == "monotype"   # sentinel untouched
    assert _base_format(None) is None
    # and it lands on the instance the chaos/ps/replay lookups key off
    assert Gen9Translator(set_source="gen9oulongtimer")._set_source == "gen9ou"


def test_late_game_world_collapse():
    from types import SimpleNamespace as NS
    from showdown.gen9_player import Gen9PokeEnginePlayer as P
    stub = P.__new__(P)
    stub._set_samples, stub._verbose = 2, False
    stub._collapse_turn, stub._collapse_moves = 25, 14
    stub._collapse_mons = 5
    stub._airi = None          # prose hook off in tests (engine-beat path)

    def battle(turn, moves_per_mon):
        team = {f"m{i}": NS(moves={f"mv{j}": None for j in range(k)})
                for i, k in enumerate(moves_per_mon)}
        return NS(turn=turn, opponent_team=team)

    full = [4, 4, 3, 2, 2, 1]     # 16 revealed moves
    thin = [2, 2, 1, 1, 0, 0]     # 6 revealed moves

    assert stub._effective_samples(battle(30, full)) == 1     # collapses
    assert stub._effective_samples(battle(20, full)) == 2     # too early
    assert stub._effective_samples(battle(30, thin)) == 2     # sets ambiguous
    stub._set_samples = 1
    assert stub._effective_samples(battle(30, full)) == 1     # no-op at K=1

    # NARROW-MOVEPOOL coverage gate: a stall team reveals few DISTINCT moves
    # (the exploit probe saw 12 across a fully-played team) but most of its
    # mons have acted — we know their hand, so collapse should still fire.
    stub._set_samples = 2
    narrow = [2, 2, 2, 2, 2, 0]   # 10 moves < 14, but 5 mons have acted
    assert stub._effective_samples(battle(30, narrow)) == 1
    barely = [2, 2, 1, 0, 0, 0]   # 5 moves, only 3 acted -> still hedge
    assert stub._effective_samples(battle(30, barely)) == 2


def test_endgame_solver_gates():
    from types import SimpleNamespace as NS
    from showdown.gen9_player import Gen9PokeEnginePlayer as P
    import showdown.gen9_player as gp

    stub = P.__new__(P)
    stub._use_endgame_solver, stub._verbose = True, False
    stub._endgame_alive, stub._endgame_depth, stub._endgame_nodes = 3, 12, 10000
    stub._airi = None

    def state(**_):
        mk = lambda: NS(pokemon=[NS(terastallized=True, hp=100)])
        return NS(side_one=mk(), side_two=mk())

    def bat(turn=40, ours=True, theirs=True):
        # the gate reads battle.team / opponent_team is_terastallized, which
        # SURVIVES a faint — the engine-state flag did not (that bug locked
        # the solver out of nearly every endgame)
        mk = lambda t: {"a": NS(is_terastallized=t)}
        return NS(turn=turn, team=mk(ours), opponent_team=mk(theirs))

    gp.is_solvable_endgame = lambda st, max_total_alive=3: True
    def fake_solve(st, max_depth=12, node_budget=10000, stats=None):
        return "earthquake", "protect", 0.8
    gp.solve_endgame = fake_solve

    b = bat()
    out = stub._try_endgame_solver(b, state())
    assert out is not None
    assert out[0].side_one[0].move_choice == "earthquake"
    assert out[0].side_one[0].visits == 1_000_000

    # tera still PENDING on either side voids the guarantee
    assert stub._try_endgame_solver(bat(ours=False), state()) is None
    assert stub._try_endgame_solver(bat(theirs=False), state()) is None

    # budget-exhausted solves are distrusted; two strikes stop the solver
    def exhausted(st, max_depth=12, node_budget=10000, stats=None):
        if stats is not None:
            stats["budget_exhausted"] = True
        return "earthquake", "protect", 0.0
    gp.solve_endgame = exhausted
    b2 = bat()
    assert stub._try_endgame_solver(b2, state()) is None
    assert b2._cb_solver_strikes == 1
    assert stub._try_endgame_solver(b2, state()) is None
    assert b2._cb_solver_strikes == 2
    calls = []
    gp.is_solvable_endgame = lambda st, max_total_alive=3: calls.append(1) or True
    assert stub._try_endgame_solver(b2, state()) is None
    assert not calls          # struck out: solver no longer consulted

    # master switch off
    stub._use_endgame_solver = False
    assert stub._try_endgame_solver(bat(), state()) is None


def test_scouting_book_is_the_top_set_tier():
    """The book is direct evidence about THIS opponent, so it outranks the
    corpus tiers — but only when we've seen the species enough times, and it
    must never overwrite what's revealed this game."""
    tr = Gen9Translator(set_source="gen9ou")

    profile = {
        "games": 9,
        "sets": {
            "Great Tusk": {
                "moves": {"headlongrush": 7, "icespinner": 6, "rapidspin": 5,
                          "knockoff": 4, "stealthrock": 1},
                "items": {"boosterenergy": 7},
                "abilities": {"protosynthesis": 7},
                "tera": {"steel": 3},
            },
            "Cinderace": {          # thin: seen once, below the gate
                "moves": {"pyroball": 1},
                "items": {}, "abilities": {}, "tera": {},
            },
        },
    }
    tr.set_opponent_book(profile, min_obs=2)

    booked = tr._book_set("greattusk")
    assert booked["item"] == "boosterenergy"
    assert booked["ability"] == "protosynthesis"
    assert booked["tera"] == "steel"
    assert booked["moves"][:4] == ["headlongrush", "icespinner",
                                   "rapidspin", "knockoff"]
    # a move revealed THIS game always survives, book fills the remainder
    revealed = tr._book_set("greattusk", known_moves=("bulldoze",))
    assert revealed["moves"][0] == "bulldoze"
    assert "headlongrush" in revealed["moves"]
    assert len(revealed["moves"]) == 4

    # count gate: one sighting is an anecdote, not a pattern
    assert tr._book_set("cinderace") is None
    # unknown species / no book at all -> fall through to the corpus tiers
    assert tr._book_set("dragapult") is None
    tr.set_opponent_book(None)
    assert tr._book_set("greattusk") is None


def test_scouting_book_reaches_opp_set():
    """End-to-end: the override actually lands in the composed set."""
    tr = Gen9Translator(set_source="gen9ou")
    base = tr._opp_set("greattusk")
    assert base is not None
    tr.set_opponent_book({"games": 5, "sets": {"Great Tusk": {
        "moves": {"bodypress": 5, "irondefense": 5, "rest": 4, "sleeptalk": 4},
        "items": {"leftovers": 5}, "abilities": {}, "tera": {}}}}, min_obs=2)
    booked = tr._opp_set("greattusk")
    assert booked["item"] == "leftovers"
    assert set(booked["moves"]) == {"bodypress", "irondefense", "rest", "sleeptalk"}
    # spread still comes from the statistical baseline (we never see EVs)
    assert booked["evs"] == base["evs"]


def test_opponent_priors_from_book():
    """Priors bias ONLY the opponent, and tera suppression is data-driven."""
    from types import SimpleNamespace as NS
    from showdown.gen9_player import Gen9PokeEnginePlayer as P
    stub = P.__new__(P)

    # a never-tera opponent (0 teras in 25 games) with a booked Kingambit
    stub._opp_profile = {
        "games": 25, "tera_turns": [],
        "sets": {"Kingambit": {"moves": {"kowtowcleave": 10, "suckerpunch": 8,
                                         "ironhead": 4, "swordsdance": 2}}},
    }
    battle = NS(opponent_active_pokemon=NS(species="Kingambit"))
    probs, suppress = stub._opponent_priors(battle)
    assert suppress is True                      # 0/25 teras -> suppress
    assert probs["kowtowcleave"] > probs["swordsdance"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6

    opts = [NS(move_choice="kowtowcleave"), NS(move_choice="swordsdance"),
            NS(move_choice="ironhead-tera"), NS(move_choice="switch gholdengo")]
    pri = stub._aligned_opp_priors(opts, probs, suppress)
    assert abs(sum(pri) - 1.0) < 1e-6
    assert pri[0] > pri[1]                       # frequent move outranks rare
    assert pri[2] < min(pri[0], pri[1], pri[3])  # tera line crushed

    # an opponent who DOES tera keeps its tera branches alive
    _, suppress2 = stub._opponent_priors.__func__(
        NS(_opp_profile={"games": 10, "tera_turns": [5, 6, 7, 8],
                         "sets": {"Kingambit": {"moves": {"ironhead": 3}}}}),
        battle)
    assert suppress2 is False

    # unknown opponent -> no priors at all (corpus behaviour unchanged)
    stub._opp_profile = None
    assert stub._opponent_priors(battle) is None


# --- stable slot ordering (cross-turn tree reuse depends on this) ----------
#
# The engine identifies the active mon via active_index, so the array does NOT
# need the active first. Slots must stay FIXED for the whole battle: a
# retained MCTS subtree's MoveChoice::Switch(i) is a slot index, so if the
# array permuted on a switch, slot i would silently denote a different mon and
# every switch line in the reused tree would be wrong. Measured on real ladder
# games, 61% of turns contain a switch, so an active-first array made reuse
# impossible on most turns (live hit rate 31% -> 92% after this change).

def _slots_of(side):
    return {p.id: i for i, p in enumerate(side.pokemon) if p.id != "none"}


def test_slots_stay_fixed_across_a_switch():
    b = make_battle()
    tr = Gen9Translator(set_source="gen9ou")
    before = _slots_of(tr.translate(b).side_one)

    b.parse_message(["", "switch", "p1a: Heatran", "Heatran, L100, M", "386/386"])
    b.parse_message(["", "turn", "2"])
    after = _slots_of(tr.translate(b).side_one)

    shared = set(before) & set(after)
    assert shared, "expected overlapping species across the switch"
    for species in shared:
        assert before[species] == after[species], (
            f"{species} moved slot {before[species]} -> {after[species]}; "
            "a permuted array corrupts switch indices in a reused subtree")


def test_active_index_tracks_the_active_not_slot_zero():
    b = make_battle()
    tr = Gen9Translator(set_source="gen9ou")
    state = tr.translate(b)
    assert state.side_one.pokemon[int(state.side_one.active_index)].id == "ninetales"

    b.parse_message(["", "switch", "p1a: Volcarona", "Volcarona, L100, M", "391/391"])
    b.parse_message(["", "turn", "2"])
    state = tr.translate(b)
    idx = int(state.side_one.active_index)
    assert state.side_one.pokemon[idx].id == "volcarona"
    assert idx != 0, "Volcarona is not the first team member; slot 0 would be stale"


def test_switch_options_exclude_the_active_after_reorder():
    """The engine derives switches from active_index; a mismatch would offer
    switching into the mon already on the field."""
    b = make_battle()
    b.parse_message(["", "switch", "p1a: Volcarona", "Volcarona, L100, M", "391/391"])
    b.parse_message(["", "turn", "2"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    result = pe.monte_carlo_tree_search(state, 50)
    switches = {r.move_choice[7:] for r in result.side_one
                if r.move_choice.startswith("switch ")}
    assert "volcarona" not in switches
    assert switches, "expected the bench to be switchable"


def test_encore_move_slot_resolves_against_the_active_mon():
    """last_used_move indexes into the ACTIVE mon's movelist. Reading slot 0
    instead would resolve against the wrong Pokemon, and the engine panics on
    an ENCORE volatile carrying a bad move slot."""
    b = make_battle()
    b.parse_message(["", "switch", "p1a: Heatran", "Heatran, L100, M", "386/386"])
    b.parse_message(["", "turn", "2"])
    b.parse_message(["", "move", "p1a: Heatran", "Taunt", "p2a: Garchomp"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    lum = state.side_one.last_used_move
    assert lum.startswith("move:"), lum
    idx = int(lum.split(":")[1])
    active = state.side_one.pokemon[int(state.side_one.active_index)]
    assert active.id == "heatran"
    assert active.moves[idx].id == "taunt"
