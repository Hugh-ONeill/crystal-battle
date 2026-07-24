"""Stall-mode: wall-war detection from pastes and the per-battle engine
flag's effect on evaluation (synergy terms + doubled recovery-PP tax)."""

import unittest

import poke_engine as pe

from showdown.gen9_player import wall_mons

STALL_PASTE = "\n\n".join(
    f"{name} @ Leftovers\nAbility: {ab}\n- {mv}\n- Protect"
    for name, ab, mv in [
        ("Clodsire", "Water Absorb", "Recover"),
        ("Clefable", "Magic Guard", "Moonlight"),
        ("Corviknight", "Pressure", "Roost"),
        ("Toxapex", "Regenerator", "Recover"),
        ("Dondozo", "Unaware", "Rest"),
        ("Gliscor", "Poison Heal", "Knock Off"),
    ])

OFFENSE_PASTE = "\n\n".join(
    f"{name} @ Life Orb\nAbility: X\n- Swords Dance\n- Earthquake"
    for name in ["Kingambit", "Iron Valiant", "Kommo-o",
                 "Kyurem", "Iron Treads", "Iron Crown"])


class TestWallDetection(unittest.TestCase):
    def test_counts(self):
        # 5 recovery movers + Poison Heal Gliscor counts via ability
        self.assertEqual(wall_mons(STALL_PASTE), 6)
        self.assertEqual(wall_mons(OFFENSE_PASTE), 0)
        self.assertEqual(wall_mons(None), 0)
        self.assertEqual(wall_mons(""), 0)

    def test_ability_healers_count_without_recovery_moves(self):
        paste = ("Slowking-Galar @ Choice Scarf\nAbility: Regenerator\n"
                 "- Trick\n- Sludge Bomb")
        self.assertEqual(wall_mons(paste), 1)

    def test_drains_do_not_count(self):
        paste = "Venusaur @ Leftovers\nAbility: Chlorophyll\n- Giga Drain"
        self.assertEqual(wall_mons(paste), 0)

    def test_real_breadth_teams(self):
        stall_b = open("showdown/teams/breadth/stall_bliss/"
                       "19_blissey_c61578fd.txt").read()
        offense = open("showdown/teams/breadth/offense/"
                       "07_kommoo_offense.txt").read()
        self.assertGreaterEqual(wall_mons(stall_b), 4)
        self.assertLess(wall_mons(offense), 4)


class TestEngineFlag(unittest.TestCase):
    def tearDown(self):
        pe.set_stall_mode(False)

    def test_flag_applies_context_weights(self):
        """Since the synergy promotion (facts always-on), the mode's
        observable effect is the context weights — here, TOXIC_ON_WALL
        repricing a badly poisoned recovery-carrying mon."""
        import json
        recs = (json.loads(l) for l in
                open("showdown/bench/pool_positions.jsonl"))
        state = None
        ability_exempt = {"POISONHEAL", "MAGICGUARD", "GUTS", "MARVELSCALE",
                          "QUICKFEET", "TOXICBOOST"}
        recovery = {"RECOVER", "ROOST", "MOONLIGHT", "SOFTBOILED",
                    "SLACKOFF", "REST"}
        for r in recs:
            s = pe.State.from_string(r["state"])
            for side in (s.side_one, s.side_two):
                for p in side.pokemon:
                    if (str(p.status) == "TOXIC"
                            and str(p.ability) not in ability_exempt
                            and any(str(m.id) in recovery for m in p.moves)):
                        state = r["state"]
            if state:
                break
        self.assertIsNotNone(state, "no tox'd wall state in pool")
        base = pe.evaluate(pe.State.from_string(state))
        pe.set_stall_mode(True)
        on = pe.evaluate(pe.State.from_string(state))
        pe.set_stall_mode(False)
        off_again = pe.evaluate(pe.State.from_string(state))
        self.assertNotAlmostEqual(base, on, places=2)
        self.assertAlmostEqual(base, off_again, places=4)


if __name__ == "__main__":
    unittest.main()
