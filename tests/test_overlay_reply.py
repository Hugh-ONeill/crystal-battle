"""Reply channel (§5.2) shadow parts: option-set assembly, prediction
validation, prompt exposure, and the ground-truth log parser the accuracy
gate scores against. The LLM call is not tested (network); the guards are —
an unvalidated action string reaching reply_pred, or a forced replacement
scored as a decision, would poison the exact audit the channel's gate is."""

from types import SimpleNamespace

from showdown.overlay import OverlayShadow, _canon_opt
from showdown.reply_audit import parse_logs, engine_dist, collapse

ROLES = {"kingambit": {"fact": "cleaner."}}


def overlay():
    return OverlayShadow(roles=ROLES)


def rr(*pairs):
    return [SimpleNamespace(move_choice=c, visits=v, total_score=0.0)
            for c, v in pairs]


def world2(*pairs):
    return SimpleNamespace(side_two=rr(*pairs))


# ---- canonicalization -------------------------------------------------

def test_canon_opt_matches_model_echoes():
    assert _canon_opt("Switch Heatran") == _canon_opt("switch heatran")
    assert _canon_opt("U-turn") == _canon_opt("uturn")
    assert _canon_opt("Earthquake-Tera") == "earthquake-tera"
    assert _canon_opt(" Stealth Rock ") == "stealthrock"


# ---- option-set assembly ----------------------------------------------

def test_reply_options_union_ordered_by_summed_share():
    res = [world2(("uturn", 600), ("stealthrock", 300), ("none", 100)),
           world2(("switch heatran", 700), ("uturn", 300))]
    opts = OverlayShadow._reply_options(res)
    assert "none" not in opts
    # uturn: 0.6 + 0.3 = 0.9 beats switch heatran 0.7 beats rock 0.3
    assert opts == ["uturn", "switch heatran", "stealthrock"]


# ---- validation -------------------------------------------------------

def test_validate_reply_filters_and_renormalizes():
    o = overlay()
    got = o._validate(
        {"world_weights": {"0": 1.0, "1": 1.0},
         "reply": {"U-turn": 0.6, "Switch Heatran": 0.2,
                   "madeupmove": 0.2}},
        2, options=["uturn", "switch heatran", "stealthrock"])
    assert got["reply_dropped"] == 1
    assert set(got["reply"]) == {"uturn", "switch heatran"}
    assert abs(sum(got["reply"].values()) - 1.0) < 1e-6
    assert got["reply"]["uturn"] > got["reply"]["switch heatran"]


def test_bad_reply_never_fails_the_consult():
    o = overlay()
    got = o._validate({"world_weights": {"0": 1.0, "1": 1.0},
                       "reply": {"nonsense": 1.0}},
                      2, options=["uturn"])
    assert got is not None and got["reply"] is None
    assert got["reply_dropped"] == 1
    got = o._validate({"world_weights": {"0": 1.0, "1": 1.0},
                       "reply": "garbage"}, 2, options=["uturn"])
    assert got is not None and got["reply"] is None
    # no options offered (old-format caller): reply is skipped, not fatal
    got = o._validate({"world_weights": {"0": 1.0, "1": 1.0},
                       "reply": {"uturn": 1.0}}, 2)
    assert got is not None and got["reply"] is None


def test_turn_message_lists_their_options():
    rec = {"turn": 5, "reasons": ["near-tie"], "engine_choice": "uturn",
           "engine_margin": 0.01, "worlds": [], "appendix": {},
           "reply_options": ["uturn", "switch heatran"]}
    msg = OverlayShadow._turn_message(rec)
    assert "THEIR OPTIONS" in msg
    assert "switch heatran" in msg
    assert "predict their click" in msg


# ---- ground-truth parser ----------------------------------------------

LOG = """\
>battle-gen9oulongtimer-111
|player|p1|PAC-Crystal|x|
|player|p2|richwoman|y|
|turn|1
|move|p1a: Glimmora|Stealth Rock|p2a: Ting-Lu
|move|p2a: Ting-Lu|Whirlwind|p1a: Glimmora
|turn|2
|switch|p2a: Goldy|Gholdengo, L50|100/100
|move|p1a: Glimmora|Mortal Spin|p2a: Goldy
|turn|3
|move|p1a: Glimmora|Power Gem|p2a: Goldy
|faint|p2a: Goldy
|switch|p2a: Ting-Lu|Ting-Lu, L50|80/100
|turn|4
|drag|p2a: Blissey|Blissey, F|100/100
|turn|5
|move|p2a: Blissey|Soft-Boiled|p2a: Blissey
|switch|p2a: Ting-Lu|Ting-Lu, L50|80/100
"""


def test_parser_decisions_and_exclusions(tmp_path):
    p = tmp_path / "overnight_x_ladder.log"
    p.write_text(LOG)
    decisions, opp = parse_logs([str(p)], "PAC-Crystal")
    assert opp["battle-gen9oulongtimer-111"] == "richwoman"
    tag = "battle-gen9oulongtimer-111"
    assert decisions[(tag, 1)] == "whirlwind"
    # hard switch is a decision; nickname ignored, details species used
    assert decisions[(tag, 2)] == "switch gholdengo"
    # fainted before acting -> replacement switch is NOT a decision
    assert (tag, 3) not in decisions
    # phazed in by us -> not a decision
    assert (tag, 4) not in decisions
    # first event rule: the move is the click, trailing switch ignored
    assert decisions[(tag, 5)] == "softboiled"


def test_engine_dist_equal_vote_and_tera_collapse():
    rec = {"worlds": [
        {"their_replies": [["uturn", 800], ["earthquake-tera", 200]]},
        {"their_replies": [["earthquake", 500], ["uturn", 500]]}]}
    d = engine_dist(rec)
    assert abs(d["uturn"] - (0.8 + 0.5) / 2) < 1e-9
    assert abs(d["earthquake"] - (0.2 + 0.5) / 2) < 1e-9
    assert collapse("gigatonhammer-tera") == "gigatonhammer"
