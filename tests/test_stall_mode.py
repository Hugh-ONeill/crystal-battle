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
        self.assertEqual(wall_mons(STALL_PASTE), 5)   # Gliscor has no recovery
        self.assertEqual(wall_mons(OFFENSE_PASTE), 0)
        self.assertEqual(wall_mons(None), 0)
        self.assertEqual(wall_mons(""), 0)

    def test_drains_do_not_count(self):
        paste = "Venusaur @ Leftovers\nAbility: Chlorophyll\n- Giga Drain"
        self.assertEqual(wall_mons(paste), 0)


class TestEngineFlag(unittest.TestCase):
    def tearDown(self):
        pe.set_stall_mode(False)

    def test_flag_activates_synergy_terms(self):
        import json
        recs = (json.loads(l) for l in
                open("showdown/bench/pool_positions.jsonl"))
        state = None
        for r in recs:
            s = pe.State.from_string(r["state"])
            if any(str(p.ability) == "REGENERATOR" and 0 < p.hp < p.maxhp
                   for p in s.side_one.pokemon):
                state = r["state"]
                break
        self.assertIsNotNone(state, "no damaged-Regenerator state in pool")
        base = pe.evaluate(pe.State.from_string(state))
        pe.set_stall_mode(True)
        on = pe.evaluate(pe.State.from_string(state))
        pe.set_stall_mode(False)
        off_again = pe.evaluate(pe.State.from_string(state))
        self.assertNotAlmostEqual(base, on, places=2)
        self.assertAlmostEqual(base, off_again, places=4)


if __name__ == "__main__":
    unittest.main()
