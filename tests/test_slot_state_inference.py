"""Substitute HP estimation, Outrage-class rampage locks, and two-turn
charge commitment (2026-08-04, fp cross-audit residuals). Obs level drives
the REAL scanner over crafted protocol; translator level checks the state
the engine actually receives."""
import logging
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator
from showdown.set_inference import BattleObservations

from tests.test_gen9_translator import REQUEST


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


def _battle(*extra):
    b = Battle("battle-gen9monotype-test-2", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F",
                     "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M",
                     "100/100"])
    b.parse_message(["", "turn", "1"])
    for e in extra:
        b.parse_message(e)
    return b


# ---- substitute hit bookkeeping (obs) -----------------------------------

SUB_STORY = [
    "|switch|p2a: Garchomp|Garchomp, M|100/100",
    "|-start|p2a: Garchomp|Substitute",
    "|move|p1a: Ninetales|Flamethrower|p2a: Garchomp",
    "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    "|turn|3",
]


def test_sub_hit_recorded_with_attacker_and_move():
    o = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY)
    assert o.opp_sub_hits() == [("ninetales", "flamethrower")]


def test_new_sub_and_broken_sub_clear_the_ledger():
    o = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY
             + ["|-start|p2a: Garchomp|Substitute"])
    assert o.opp_sub_hits() == []
    o2 = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY
              + ["|-end|p2a: Garchomp|Substitute"])
    assert o2.opp_sub_hits() == []


def test_switch_clears_sub_hits():
    o = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY
             + ["|switch|p2a: Kingambit|Kingambit, M|100/100"])
    assert o.opp_sub_hits() == []


# ---- substitute HP estimate (translator) --------------------------------

def test_their_hit_sub_is_estimated_below_full():
    # translate() itself replays the protocol into obs, so the whole chain
    # (activation -> attacker pairing -> engine-calc estimate) runs live
    b = _battle(["", "-start", "p2a: Garchomp", "Substitute"],
                ["", "move", "p1a: Ninetales", "Flamethrower",
                 "p2a: Garchomp"],
                ["", "-activate", "p2a: Garchomp", "move: Substitute",
                 "[damage]"])
    state = Gen9Translator().translate(b)
    sub_max = state.side_two.pokemon[0].maxhp // 4
    assert 1 <= state.side_two.substitute_health < sub_max


def test_their_fresh_sub_stays_at_quarter():
    b = _battle(["", "-start", "p2a: Garchomp", "Substitute"])
    state = Gen9Translator().translate(b)
    assert state.side_two.substitute_health == \
        state.side_two.pokemon[0].maxhp // 4


def test_our_hit_sub_uses_the_binary_read():
    b = _battle(["", "-start", "p1a: Ninetales", "Substitute"],
                ["", "move", "p2a: Garchomp", "Earthquake",
                 "p1a: Ninetales"],
                ["", "-activate", "p1a: Ninetales", "move: Substitute",
                 "[damage]"])
    state = Gen9Translator().translate(b)
    assert state.side_one.substitute_health == 323 // 10


# ---- rampage lock (obs + translator) ------------------------------------

def test_rampage_counts_consecutive_uses_and_ends_on_other_move():
    o = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Outrage|p1a: X", "|turn|2",
              "|move|p2a: Garchomp|Outrage|p1a: X", "|turn|3"])
    assert o.rampage_for("garchomp") == ("garchomp", "outrage", 2)
    o2 = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
               "|move|p2a: Garchomp|Outrage|p1a: X", "|turn|2",
               "|move|p2a: Garchomp|Earthquake|p1a: X", "|turn|3"])
    assert o2.rampage_for("garchomp") is None


def test_rampage_ends_on_fatigue_cant_and_miss():
    base = ["|switch|p2a: Garchomp|Garchomp, M|100/100",
            "|move|p2a: Garchomp|Outrage|p1a: X"]
    ends = (["|-start|p2a: Garchomp|confusion|[fatigue]"],
            ["|cant|p2a: Garchomp|slp"],
            ["|-miss|p2a: Garchomp|p1a: X"])
    for tail in ends:
        assert _obs(base + tail + ["|turn|2"]).rampage_for("garchomp") is None


def test_rampaging_opponent_is_pinned_to_the_move():
    b = _battle(["", "move", "p2a: Garchomp", "Outrage", "p1a: Ninetales"])
    tr = Gen9Translator()
    state = tr.translate(b)
    chomp = state.side_two.pokemon[0]
    by_id = {m.id: m for m in chomp.moves if m.id != "none"}
    assert by_id["outrage"].disabled is False
    assert all(m.disabled for mid, m in by_id.items() if mid != "outrage")
    assert "lockedmove" in state.side_two.volatile_statuses
    assert state.side_two.volatile_status_durations.lockedmove == 1


# ---- charge commitment (obs + translator) -------------------------------

def test_charge_set_on_prepare_cleared_on_release():
    charge = ["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Meteor Beam|p1a: X|[still]",
              "|-prepare|p2a: Garchomp|Meteor Beam"]
    o = _obs(charge)
    assert o.opp_charging() == "meteorbeam"
    o2 = _obs(charge + ["|turn|2",
                        "|move|p2a: Garchomp|Meteor Beam|p1a: X"])
    assert o2.opp_charging() is None


def test_power_herb_release_clears_the_charge():
    o = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Meteor Beam|p1a: X|[still]",
              "|-prepare|p2a: Garchomp|Meteor Beam",
              "|-enditem|p2a: Garchomp|Power Herb"])
    assert o.opp_charging() is None


def test_charging_opponent_carries_volatile_and_pin():
    b = _battle(["", "move", "p2a: Garchomp", "Solar Beam", "p1a: Ninetales",
                 "[still]"],
                ["", "-prepare", "p2a: Garchomp", "Solar Beam"])
    state = Gen9Translator().translate(b)
    assert "solarbeam" in state.side_two.volatile_statuses
    chomp = state.side_two.pokemon[0]
    by_id = {m.id: m for m in chomp.moves if m.id != "none"}
    assert by_id["solarbeam"].disabled is False
    assert all(m.disabled for mid, m in by_id.items() if mid != "solarbeam")


# ---- spread-level reverse damage calc (suspect #2, 2026-08-05) ----------

def _max_roll(att, defd, move):
    import poke_engine as pe
    st = pe.State(
        side_one=pe.Side(pokemon=[att] + [
            pe.Pokemon.create_fainted() for _ in range(5)]),
        side_two=pe.Side(pokemon=[defd] + [
            pe.Pokemon.create_fainted() for _ in range(5)]))
    return max(pe.calculate_damage(st, move, "splash", True)[0])


def _cand(nature, atk_evs, item):
    evs = {"hp": 0, "atk": atk_evs, "def": 0, "spa": 0, "spd": 0, "spe": 252}
    ivs = {k: 31 for k in evs}
    return {"nature": nature, "evs": evs, "ivs": ivs, "item": item,
            "ability": "roughskin", "moves": ["earthquake"]}


def test_spread_ruled_out_separates_banded_from_bulky():
    from showdown.gen9_translator import Gen9Translator
    from tests.test_gen9_translator import make_battle
    tr = Gen9Translator()
    tr.translate(make_battle())          # populates _my_built (real stats)
    ours = tr._my_built["ninetales"]

    banded = tr._probe_from_set("garchomp", _cand("Jolly", 252, "choiceband"))
    bulky = tr._probe_from_set("garchomp", _cand("Impish", 0, "leftovers"))
    hi, lo = _max_roll(banded, ours, "earthquake"), _max_roll(bulky, ours,
                                                              "earthquake")
    assert hi > lo * 1.3, "test premise: builds must separate cleanly"

    obs = BattleObservations()
    obs.damage_evidence.append({
        "species": "garchomp", "move": "earthquake",
        "damage": (hi + lo) // 2, "our_species": "ninetales",
        "weather": "none", "se": False})
    assert obs.spread_ruled_out(bulky, tr._my_built) is True
    assert obs.spread_ruled_out(banded, tr._my_built) is False


def test_under_max_hit_never_convicts():
    """Screens/Multiscale/burn deflate damage and are ungated by evidence
    curation, so a weak observed hit must not rule out strong builds."""
    from showdown.gen9_translator import Gen9Translator
    from tests.test_gen9_translator import make_battle
    tr = Gen9Translator()
    tr.translate(make_battle())
    banded = tr._probe_from_set("garchomp", _cand("Jolly", 252, "choiceband"))
    obs = BattleObservations()
    obs.damage_evidence.append({
        "species": "garchomp", "move": "earthquake", "damage": 5,
        "our_species": "ninetales", "weather": "none", "se": False})
    assert obs.spread_ruled_out(banded, tr._my_built) is False


# ---- defensive reverse calc: our hits on them (2026-08-05) --------------

def _rolls(att, defd, move):
    import poke_engine as pe
    st = pe.State(
        side_one=pe.Side(pokemon=[att] + [
            pe.Pokemon.create_fainted() for _ in range(5)]),
        side_two=pe.Side(pokemon=[defd] + [
            pe.Pokemon.create_fainted() for _ in range(5)]))
    return pe.calculate_damage(st, move, "splash", True)[0]


def _dcand(nature, hp_evs, spd_evs):
    evs = {"hp": hp_evs, "atk": 0, "def": 0, "spa": 0, "spd": spd_evs,
           "spe": 0}
    return {"nature": nature, "evs": evs, "ivs": {k: 31 for k in evs},
            "item": "none", "ability": "unaware", "moves": ["recover"]}


def _armed(tr, delta_events):
    from tests.test_gen9_translator import make_battle
    b = make_battle()
    for e in delta_events:
        b.parse_message(e)
    tr.translate(b)
    return tr._obs


def test_our_hit_evidence_captured_with_gates():
    from showdown.gen9_translator import Gen9Translator
    tr = Gen9Translator()
    obs = _armed(tr, [
        ["", "-sidestart", "p2: opp", "Reflect"],
        ["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Garchomp"],
        ["", "-damage", "p2a: Garchomp", "70/100"], ["", "turn", "2"]])
    ev = obs.our_damage_evidence[-1]
    assert ev["delta"] == 30 and ev["their_species"] == "garchomp"
    assert ev["screens"] == ("reflect",) and ev["truncated"] is False
    tr2 = Gen9Translator()
    obs2 = _armed(tr2, [
        ["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Garchomp"],
        ["", "-crit", "p2a: Garchomp"],
        ["", "-damage", "p2a: Garchomp", "50/100"], ["", "turn", "2"]])
    assert not obs2.our_damage_evidence     # crit invalidates the read


def test_bulk_check_separates_frail_from_bulky():
    from showdown.gen9_translator import Gen9Translator
    tr = Gen9Translator()
    tr.translate(__import__("tests.test_gen9_translator",
                            fromlist=["make_battle"]).make_battle())
    ours = tr._my_built["ninetales"]
    frail = tr._probe_from_set("clodsire", _dcand("Serious", 0, 0))
    bulky = tr._probe_from_set("clodsire", _dcand("Calm", 252, 252))
    rf, rb = _rolls(ours, frail, "flamethrower"), _rolls(ours, bulky,
                                                        "flamethrower")
    mid_frail_pct = round(100 * (sum(rf) / len(rf)) / frail.maxhp)
    obs_abs = mid_frail_pct / 100
    assert obs_abs * bulky.maxhp > max(rb) * 1.025 + 5, \
        "premise: the observation must exceed the bulky candidate's band"
    assert (min(rf) * 0.975 - 5 < obs_abs * frail.maxhp
            < max(rf) * 1.025 + 5), \
        "premise: the observation must sit inside the frail candidate's band"

    tr2 = Gen9Translator()
    obs = _armed(tr2, [
        ["", "switch", "p2a: Clodsire", "Clodsire, M", "100/100"],
        ["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Clodsire"],
        ["", "-damage", "p2a: Clodsire", f"{100 - mid_frail_pct}/100"],
        ["", "turn", "2"]])
    assert obs.bulk_ruled_out(bulky, tr._my_built) is True
    assert obs.bulk_ruled_out(frail, tr._my_built) is False


def test_bulk_lower_violation_and_ko_truncation():
    from showdown.gen9_translator import Gen9Translator
    tr = Gen9Translator()
    tr.translate(__import__("tests.test_gen9_translator",
                            fromlist=["make_battle"]).make_battle())
    ours = tr._my_built["ninetales"]
    frail = tr._probe_from_set("clodsire", _dcand("Serious", 0, 0))
    bulky = tr._probe_from_set("clodsire", _dcand("Calm", 252, 252))
    rb = _rolls(ours, bulky, "flamethrower")
    mid_bulky_pct = max(1, round(100 * (sum(rb) / len(rb)) / bulky.maxhp))

    tr2 = Gen9Translator()
    obs = _armed(tr2, [
        ["", "switch", "p2a: Clodsire", "Clodsire, M", "100/100"],
        ["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Clodsire"],
        ["", "-damage", "p2a: Clodsire", f"{100 - mid_bulky_pct}/100"],
        ["", "turn", "2"]])
    assert obs.bulk_ruled_out(frail, tr._my_built) is True

    tr3 = Gen9Translator()
    obs3 = _armed(tr3, [
        ["", "switch", "p2a: Clodsire", "Clodsire, M", "100/100"],
        ["", "-damage", "p2a: Clodsire", "3/100", "[from] Stealth Rock"],
        ["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Clodsire"],
        ["", "-damage", "p2a: Clodsire", "0 fnt"],
        ["", "faint", "p2a: Clodsire"], ["", "turn", "2"]])
    assert obs3.bulk_ruled_out(frail, tr._my_built) is False   # truncated


def test_multiscale_gate_blocks_full_hp_conviction_only():
    from showdown.gen9_translator import Gen9Translator
    tr = Gen9Translator()
    tr.translate(__import__("tests.test_gen9_translator",
                            fromlist=["make_battle"]).make_battle())
    frail_dnite = tr._probe_from_set("dragonite", _dcand("Serious", 0, 0))
    for prev_events, expect in (
            ([], False),                                     # full HP: gated
            ([["", "-damage", "p2a: Dragonite", "80/100",
               "[from] Stealth Rock"]], True)):              # chipped: real
        tr_n = Gen9Translator()
        obs = _armed(tr_n, [
            ["", "switch", "p2a: Dragonite", "Dragonite, M", "100/100"],
            *prev_events,
            ["", "move", "p1a: Ninetales", "Flamethrower", "p2a: Dragonite"],
            ["", "-damage", "p2a: Dragonite",
             f"{78 if prev_events else 98}/100"],
            ["", "turn", "2"]])
        assert obs.bulk_ruled_out(frail_dnite, tr._my_built) is expect


# ---- offensive LOWER bound (2026-08-05, unblocked by screen tracking) ---

def _their_hit_obs(observed, pre=(), our_hp="323/323", move="Tackle"):
    # Tackle, deliberately: a hit weak enough to leave a survivor — a
    # Banded EQ mid-roll exceeds Ninetales' whole HP bar, and a crafted
    # "hit" that would have KO'd is (correctly) refused as truncated
    start = int(our_hp.split("/")[0])
    return _obs(["|switch|p1a: Ninetales|Ninetales, F|" + our_hp,
                 "|switch|p2a: Garchomp|Garchomp, M|100/100",
                 *pre,
                 f"|move|p2a: Garchomp|{move}|p1a: Ninetales",
                 f"|-damage|p1a: Ninetales|{max(0, start - observed)}/323"
                 if start - observed > 0 else
                 "|-damage|p1a: Ninetales|0 fnt",
                 "|turn|2"])


def _atk_probes(tr):
    band = tr._probe_from_set("garchomp", _cand("Jolly", 252, "choiceband"))
    weak = tr._probe_from_set("garchomp", _cand("Bold", 0, "leftovers"))
    return band, weak


def test_lower_bound_separates_band_from_uninvested():
    from showdown.gen9_translator import Gen9Translator
    from tests.test_gen9_translator import make_battle
    tr = Gen9Translator()
    tr.translate(make_battle())
    ours = tr._my_built["ninetales"]
    band, weak = _atk_probes(tr)
    rb, rw_ = _rolls(band, ours, "tackle"), _rolls(weak, ours, "tackle")
    observed = round(sum(rw_) / len(rw_))
    assert observed < min(rb) * 0.975 - 5, "premise: weak mid below Band min"
    assert observed < 323, "premise: the survivor must survive"
    o = _their_hit_obs(observed)
    our_mons = {"ninetales": ours}
    assert o.spread_ruled_out(band, our_mons) is True
    assert o.spread_ruled_out(weak, our_mons) is False


def test_lower_bound_respects_our_reflect():
    from showdown.gen9_translator import Gen9Translator
    from tests.test_gen9_translator import make_battle
    tr = Gen9Translator()
    tr.translate(make_battle())
    ours = tr._my_built["ninetales"]
    band, _ = _atk_probes(tr)
    rb = _rolls(band, ours, "tackle")
    observed = round(sum(rb) / len(rb) / 2)     # a Band hit THROUGH Reflect
    assert observed < 323
    o = _their_hit_obs(observed,
                       pre=("|-sidestart|p1: Us|move: Reflect",))
    assert o.spread_ruled_out(band, {"ninetales": ours}) is False


def test_lower_bound_gates_on_burn_stage_and_truncation():
    from showdown.gen9_translator import Gen9Translator
    from tests.test_gen9_translator import make_battle
    tr = Gen9Translator()
    tr.translate(make_battle())
    ours = tr._my_built["ninetales"]
    band, weak = _atk_probes(tr)
    observed = round(sum(_rolls(weak, ours, "tackle")) / 16)
    our_mons = {"ninetales": ours}
    for pre in (("|-status|p2a: Garchomp|brn",),
                ("|-unboost|p2a: Garchomp|atk|1",)):
        assert _their_hit_obs(observed, pre=pre).spread_ruled_out(
            band, our_mons) is False, pre
    # KO from low HP: damage-at-least, never convicts the lower bound
    ko = _their_hit_obs(60, our_hp="60/323")
    assert ko.spread_ruled_out(band, our_mons) is False
