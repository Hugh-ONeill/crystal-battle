"""Unit tests for the tree-reuse wiring in gen9_player: the engine-option
normalizer and the opponent-action reconstructor. Both feed MctsHandle.advance,
where a wrong answer would advance the tree through a transition that never
happened — every ambiguous case must come back None (reset, never guess)."""

from types import SimpleNamespace

from showdown.gen9_player import Gen9PokeEnginePlayer, _norm_opt


def _battle(events, role="p1"):
    return SimpleNamespace(_replay_data=events, player_role=role)


def _opp_action(events, snap=0, role="p1"):
    # _opp_action_since only touches the battle argument, never self
    return Gen9PokeEnginePlayer._opp_action_since(
        None, _battle(events, role), snap)


class TestNormOpt:
    def test_plain_move(self):
        assert _norm_opt("knockoff") == "knockoff"

    def test_switch_keeps_prefix(self):
        assert _norm_opt("switch Grimmsnarl") == "switch grimmsnarl"

    def test_tera_suffix_survives_normalize(self):
        assert _norm_opt("Knock Off-tera") == "knockoff-tera"

    def test_no_move_is_nomove(self):
        assert _norm_opt("No Move") == "nomove"

    def test_forme_hyphens_stripped(self):
        assert _norm_opt("switch Landorus-Therian") == "switch landorustherian"


class TestOppActionSince:
    def test_single_move(self):
        ev = [["", "move", "p2a: Kingambit", "Knock Off", "p1a: Gliscor"]]
        assert _opp_action(ev) == "knockoff"

    def test_move_with_tera(self):
        ev = [
            ["", "-terastallize", "p2a: Kingambit", "Dark"],
            ["", "move", "p2a: Kingambit", "Sucker Punch", "p1a: Gliscor"],
        ]
        assert _opp_action(ev) == "suckerpunch-tera"

    def test_switch(self):
        ev = [["", "switch", "p2a: Lando", "Landorus-Therian, L78, M",
               "100/100"]]
        assert _opp_action(ev) == "switch landorustherian"

    def test_faint_is_ambiguous(self):
        ev = [
            ["", "move", "p2a: Kingambit", "Knock Off", "p1a: Gliscor"],
            ["", "faint", "p1a: Gliscor"],
        ]
        assert _opp_action(ev) is None

    def test_drag_is_ambiguous(self):
        ev = [
            ["", "move", "p1a: Skarmory", "Whirlwind", "p2a: Kingambit"],
            ["", "drag", "p2a: Ting-Lu", "Ting-Lu, L80", "100/100"],
        ]
        assert _opp_action(ev) is None

    def test_opp_cant_is_ambiguous(self):
        ev = [["", "cant", "p2a: Kingambit", "par"]]
        assert _opp_action(ev) is None

    def test_our_cant_is_not(self):
        ev = [
            ["", "cant", "p1a: Gliscor", "slp"],
            ["", "move", "p2a: Kingambit", "Iron Head", "p1a: Gliscor"],
        ]
        assert _opp_action(ev) == "ironhead"

    def test_two_actions_is_ambiguous(self):
        # locked-move / Sleep Talk callouts produce two |move| lines
        ev = [
            ["", "move", "p2a: Blissey", "Sleep Talk", None],
            ["", "move", "p2a: Blissey", "Soft-Boiled", "p2a: Blissey"],
        ]
        assert _opp_action(ev) is None

    def test_pivot_chain_is_ambiguous(self):
        ev = [
            ["", "move", "p2a: Slowking", "Chilly Reception", None],
            ["", "switch", "p2a: Ting-Lu", "Ting-Lu, L80", "100/100"],
        ]
        assert _opp_action(ev) is None

    def test_no_actions_is_nomove(self):
        # coming out of OUR forced switch the opponent just waited
        ev = [["", "switch", "p1a: Corviknight", "Corviknight, L80",
               "100/100"]]
        assert _opp_action(ev) == "nomove"

    def test_snap_slices_history(self):
        ev = [
            ["", "move", "p2a: Kingambit", "Knock Off", "p1a: Gliscor"],
            ["", "turn", "12"],
            ["", "move", "p2a: Kingambit", "Iron Head", "p1a: Gliscor"],
        ]
        assert _opp_action(ev, snap=2) == "ironhead"

    def test_p2_role_flips_sides(self):
        ev = [
            ["", "move", "p1a: Kingambit", "Knock Off", "p2a: Gliscor"],
            ["", "move", "p2a: Gliscor", "Toxic", "p1a: Kingambit"],
        ]
        assert _opp_action(ev, role="p2") == "knockoff"

    def test_our_moves_ignored(self):
        ev = [
            ["", "move", "p1a: Gliscor", "Toxic", "p2a: Kingambit"],
            ["", "move", "p2a: Kingambit", "Iron Head", "p1a: Gliscor"],
        ]
        assert _opp_action(ev) == "ironhead"

    def test_missing_replay_is_ambiguous(self):
        assert Gen9PokeEnginePlayer._opp_action_since(
            None, SimpleNamespace(player_role="p1"), 0) is None

    def test_stale_snap_is_ambiguous(self):
        ev = [["", "move", "p2a: Kingambit", "Knock Off", "p1a: Gliscor"]]
        assert _opp_action(ev, snap=5) is None
