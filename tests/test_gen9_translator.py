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
    assert state.side_one.pokemon[0].hp == 0  # fainted active holds slot 0
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
    b = NS(_replay_data=[
        ["", "inactive", "Battle timer is ON."],
        ["", "inactive", "Time left: 300 sec this turn | 300 sec total | 60 sec grace"],
        ["", "inactive", "CBGen9 has 270 seconds left."],
        ["", "inactive", "CBGen9 has 150 seconds left."],
    ])
    assert _time_left(b, "CBGen9") == 150       # most recent our-bank reading
    assert _time_left(NS(_replay_data=[]), "CBGen9") is None
    # falls back to 'sec total' when no per-player line matches our name
    b2 = NS(_replay_data=[
        ["", "inactive", "Time left: 200 sec this turn | 200 sec total"]])
    assert _time_left(b2, "CBGen9") == 200


def test_adaptive_escalation_decision():
    from types import SimpleNamespace as NS
    from showdown.gen9_player import Gen9PokeEnginePlayer as P

    # decisive probe (one move 90% of visits) -> no escalation regardless
    peaked = [NS(side_one=[NS(move_choice="a", visits=900, total_score=800),
                           NS(move_choice="b", visits=100, total_score=40)])]
    # flat probe (split ~50/50) -> escalate when clock healthy
    flat = [NS(side_one=[NS(move_choice="a", visits=520, total_score=300),
                         NS(move_choice="b", visits=480, total_score=270)])]

    calls = []
    stub = P.__new__(P)
    stub._probe_ms, stub._escalate_ms = 300, 2000
    stub._flat_threshold, stub._clock_floor_s = 0.55, 40
    stub._set_samples, stub._verbose = 2, False
    stub._escalate_bank_s, stub._bank_used_s = 90.0, 0.0
    stub._escalate_min_turn, stub._escalate_min_gap = 20, 8
    stub._last_escalate_turn = -999

    def fake_search(battle, ms=None, use_value=None):
        calls.append((ms, use_value))
        return battle._probe if ms == stub._probe_ms else [NS(side_one=[])]
    stub._search_samples = fake_search

    # peaked position: one search only
    calls.clear()
    b = NS(turn=20, _replay_data=[], _probe=peaked)
    b.username = "X"
    import showdown.gen9_player as gp
    gp._time_left = lambda *a: 120  # healthy clock
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=peaked))
    assert calls == [(300, False)]  # decisive -> probe only, no value

    # flat position, healthy clock: escalates
    calls.clear()
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert (300, False) in calls and (2000, True) in calls  # probe plain, escalate value

    # flat position, low clock: no escalation (safety)
    calls.clear()
    gp._time_left = lambda *a: 20
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert calls == [(300, False)]

    # flat position, healthy clock, but bank exhausted: no escalation
    calls.clear()
    gp._time_left = lambda *a: 120
    stub._bank_used_s = 90.0
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=20, _replay_data=[], _probe=flat))
    assert calls == [(300, False)]
    stub._bank_used_s = 0.0

    # opening flatness (turn < min_turn) is skipped — bank saved for the grind
    calls.clear()
    stub._last_escalate_turn = -999
    stub._adaptive_search(NS(turn=10, _replay_data=[], _probe=flat))
    assert calls == [(300, False)]

    # too soon after the last escalation (min_gap) — spread across the game
    calls.clear()
    stub._last_escalate_turn = 100
    stub._adaptive_search(NS(turn=104, _replay_data=[], _probe=flat))
    assert calls == [(300, False)]
    # far enough past the last escalation, late game: escalates
    calls.clear()
    stub._adaptive_search(NS(turn=110, _replay_data=[], _probe=flat))
    assert (300, False) in calls and (2000, True) in calls
    assert stub._last_escalate_turn == 110  # updated for the next gap check


def test_parse_engine_choice():
    assert parse_engine_choice("switch heatran") == ("switch", "heatran")
    assert parse_engine_choice("flamethrower") == ("move", "flamethrower")
