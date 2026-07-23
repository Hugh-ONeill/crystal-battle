"""Team-archive matcher: roster indexing, revealed-info consistency
filtering, correlated whole-team sampling, and clean fallbacks."""

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from showdown.team_archive import TeamArchive, build_index, roster_key

TEAM_A = """Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 12 Def / 252 Spe
Jolly Nature
- Substitute
- Toxic
- Earthquake
- Protect

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Ghost
Adamant Nature
- Swords Dance
- Iron Head
- Sucker Punch
- Kowtow Cleave
"""

# same roster, different details: Gliscor is SD Facade, Kingambit is Air Balloon
TEAM_B = TEAM_A.replace("Toxic Orb", "Toxic Orb\nAbility: Poison Heal") \
    .replace("- Substitute\n- Toxic", "- Swords Dance\n- Facade") \
    .replace("Ability: Poison Heal\nAbility: Poison Heal", "Ability: Poison Heal") \
    .replace("Kingambit @ Leftovers", "Kingambit @ Air Balloon")

# different roster entirely
TEAM_C = TEAM_A.replace("Gliscor @ Toxic Orb", "Weavile @ Heavy-Duty Boots") \
    .replace("Ability: Poison Heal", "Ability: Pressure")


def make_archive(d):
    root = Path(d) / "teams"
    (root / "x").mkdir(parents=True)
    (root / "x" / "team_1.gen9ou_team").write_text(TEAM_A)
    (root / "x" / "team_2.gen9ou_team").write_text(TEAM_B)
    (root / "x" / "team_3.gen9ou_team").write_text(TEAM_C)
    out = Path(d) / "idx.json"
    build_index(root, out, team_size=2)
    return TeamArchive(str(out))


class TestArchive(unittest.TestCase):
    def test_roster_key_order_and_form_insensitive(self):
        self.assertEqual(roster_key(["Ting-Lu", "Iron Valiant"]),
                         roster_key(["ironvaliant", "tinglu"]))

    def test_exact_match_and_miss(self):
        with TemporaryDirectory() as d:
            arch = make_archive(d)
            self.assertEqual(len(arch.candidates(["Gliscor", "Kingambit"])), 2)
            self.assertEqual(arch.candidates(["Gliscor", "Garganacl"]), [])
            self.assertIsNone(arch.sample(["Gliscor", "Garganacl"], {}))

    def test_revealed_moves_filter_selects_consistent_team(self):
        with TemporaryDirectory() as d:
            arch = make_archive(d)
            revealed = {"gliscor": {"moves": {"substitute"}}}
            for seed in range(6):
                team = arch.sample(["Gliscor", "Kingambit"], revealed,
                                   rng=random.Random(seed))
                # only TEAM_A's Gliscor has Substitute; the CORRELATED draw
                # must therefore give TEAM_A's Kingambit item too
                self.assertEqual(team["gliscor"]["item"], "toxicorb")
                self.assertEqual(team["kingambit"]["item"], "leftovers")

    def test_revealed_item_filter(self):
        with TemporaryDirectory() as d:
            arch = make_archive(d)
            revealed = {"kingambit": {"moves": set(), "item": "airballoon"}}
            team = arch.sample(["Gliscor", "Kingambit"], revealed)
            self.assertEqual(team["kingambit"]["item"], "airballoon")
            # and the correlated Gliscor comes from TEAM_B: SD Facade
            self.assertIn("facade", team["gliscor"]["moves"])

    def test_all_candidates_eliminated_returns_none(self):
        with TemporaryDirectory() as d:
            arch = make_archive(d)
            revealed = {"gliscor": {"moves": {"knockoff"}}}
            self.assertIsNone(arch.sample(["Gliscor", "Kingambit"], revealed))


if __name__ == "__main__":
    unittest.main()
