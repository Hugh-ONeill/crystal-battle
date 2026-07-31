"""Shadow-overlay pure parts: gating, validation, citation guard, flip math.

The LLM call itself is not tested here (network); everything around it is,
because the guards are the design — an unvalidated weight vector or an
unresolved citation reaching the log would poison the flip audit the whole
phase exists to produce.
"""

from types import SimpleNamespace

from showdown.overlay import OverlayShadow, _norm_path

ROLES = {
    "kingambit": {"value_curve": "grows_with_own_faints",
                  "lead_intent": "avoid", "fact": "cleaner.",
                  "sequence": ["wait", "clean"]},
    "ceruledge": {"entry_condition": "full_hp", "fact": "sash plan."},
    "pelipper": {"resource": "rain", "fact": "sets rain on entry."},
}


def overlay():
    return OverlayShadow(roles=ROLES)


def mon(species, active=False, fainted=False, hp=1.0, moves=(),
        item=None, ability=None):
    return SimpleNamespace(species=species, active=active, fainted=fainted,
                           current_hp_fraction=hp,
                           moves={m: None for m in moves},
                           item=item, ability=ability, tera_type=None)


def battle(turn=10, team=(), opp=(), weather=None):
    return SimpleNamespace(
        battle_tag="battle-gen9ou-1", turn=turn,
        team={m.species: m for m in team},
        opponent_team={m.species: m for m in opp},
        weather=weather or {}, fields={},
        side_conditions={}, opponent_side_conditions={})


def rr(*pairs):
    return [SimpleNamespace(move_choice=c, visits=v, total_score=s)
            for c, v, s in pairs]


def world(*pairs):
    return SimpleNamespace(side_one=rr(*pairs))


def test_gate_opening_and_near_tie():
    o = overlay()
    b = battle(turn=2, team=[mon("garganacl", active=True)])
    ranked = rr(("saltcure", 510, 300.0), ("recover", 490, 280.0))
    reasons = o.consult_reasons(b, ranked, [])
    assert "opening" in reasons and "near-tie" in reasons


def test_gate_weather_down_only_when_setter_benched_and_field_absent():
    o = overlay()
    peli = mon("pelipper")
    b = battle(team=[mon("barraskewda", active=True), peli])
    ranked = rr(("liquidation", 900, 500.0))
    assert any(r.startswith("weather-down") for r in
               o.consult_reasons(b, ranked, []))
    b2 = battle(team=[mon("barraskewda", active=True), peli],
                weather={"Weather.RAINDANCE": 3})
    assert not any(r.startswith("weather-down") for r in
                   o.consult_reasons(b2, ranked, []))
    peli.fainted = True
    assert not any(r.startswith("weather-down") for r in
                   o.consult_reasons(b, ranked, []))


def test_gate_cleaner_early_needs_the_switch_on_the_table():
    o = overlay()
    b = battle(team=[mon("kyurem", active=True), mon("kingambit")])
    on_table = rr(("switch kingambit", 500, 250.0), ("icebeam", 400, 220.0))
    off_table = rr(("icebeam", 800, 450.0), ("earthpower", 100, 40.0))
    assert any(r.startswith("cleaner-early") for r in
               o.consult_reasons(b, on_table, []))
    assert not any(r.startswith("cleaner-early") for r in
                   o.consult_reasons(b, off_table, []))


def test_gate_chipped_setup():
    o = overlay()
    b = battle(team=[mon("ceruledge", active=True, hp=0.8)])
    ranked = rr(("swordsdance", 700, 400.0), ("bitterblade", 300, 150.0))
    assert any(r.startswith("chipped-setup") for r in
               o.consult_reasons(b, ranked, []))
    b_full = battle(team=[mon("ceruledge", active=True, hp=1.0)])
    assert not any(r.startswith("chipped-setup") for r in
                   o.consult_reasons(b_full, ranked, []))


def test_validate_normalizes_and_fills_missing_worlds():
    o = overlay()
    got = o._validate({"world_weights": {"0": 3.0, "2": 1.0}}, 3)
    w = got["world_weights"]
    assert len(w) == 3 and abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[2] > w[1]           # unnamed world 1 gets epsilon


def test_validate_rejects_junk():
    o = overlay()
    assert o._validate({"world_weights": {"9": 1.0}}, 2) is None
    assert o._validate({"world_weights": {}}, 2) is None
    assert o._validate({"world_weights": {"0": "fast"}}, 2) is None


def test_citation_guard_drops_unresolved_rules():
    o = overlay()
    raw = {"world_weights": {"0": 1.0, "1": 1.0},
           "flags": [{"row": "x", "rule": "kingambit.value_curve"},
                     {"row": "x", "rule": "Ceruledge.entry_condition"},
                     {"row": "x", "rule": "zoroark.made_up"},
                     {"row": "x", "rule": "kingambit.hallucinated_field"}]}
    got = o._validate(raw, 2)
    assert got["dropped_flags"] == 2
    assert [f["rule"] for f in raw["flags"]] == \
        ["kingambit.value_curve", "Ceruledge.entry_condition"]


def test_norm_path():
    assert _norm_path("Great Tusk.sequence[1]") == ["greattusk", "sequence", "1"]
    assert _norm_path("") == []


def test_flips_at_full_lambda_hand_the_vote_to_the_weighted_world():
    o = overlay()
    a = world(("icebeam", 600, 360.0), ("uturn", 400, 160.0))
    b = world(("uturn", 900, 540.0), ("icebeam", 100, 30.0))
    flips = o._flips([0.999, 0.001], [a, b], engine_choice="uturn")
    assert flips["1.0"]["top"] == "icebeam" and flips["1.0"]["flip"]
    # near-uniform blend at lambda=0.25 stays with the engine's balanced vote
    flips_soft = o._flips([0.5, 0.5], [a, b], engine_choice="uturn")
    assert not flips_soft["0.25"]["flip"]


# --- advocate world: nominated starved actions get a dedicated search --------

def test_nomination_requires_both_rule_and_starvation():
    o = overlay()
    ranked_starved = rr(("hurricane", 950, 500.0), ("switch pelipper", 20, 8.0),
                        ("uturn", 30, 12.0))
    ranked_fed = rr(("hurricane", 700, 380.0), ("switch pelipper", 300, 150.0))
    noms = o._nominations(["weather-down:pelipper", "near-tie"], ranked_starved)
    assert noms == ["switch pelipper"]
    assert o._nominations(["weather-down:pelipper"], ranked_fed) == []
    assert o._nominations(["near-tie"], ranked_starved) == []


def test_advocate_priors_concentrate_without_zeroing_the_rest():
    o = overlay()
    s1 = o._advocate_priors(["a", "b", "switch pelipper", "d"],
                            "switch pelipper")
    assert abs(sum(s1) - 1.0) < 1e-9
    assert s1[2] == 0.75 and all(w > 0 for w in s1)
    assert o._advocate_priors(["a", "b"], "missing") is None
    assert o._advocate_priors(["only"], "only") is None
