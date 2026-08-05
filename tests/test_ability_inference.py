"""The set_inference cheap trio (2026-08-04, from the fp cross-audit):
AV-after-status item disproof, Regenerator proven by re-switch HP gain, and
negative ability evidence from a silent first switch-in. Drives the REAL
BattleObservations scanner over crafted protocol, boots-harness style."""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import BattleObservations


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


# ---- Assault Vest after a status move ----------------------------------

def test_selected_status_move_forbids_assault_vest():
    o = _obs(["|switch|p2a: Slowking|Slowking-Galar, M|100/100",
              "|move|p2a: Slowking|Chilly Reception|p1a: X",
              "|turn|2"])
    assert "assaultvest" in o.forbidden("slowkinggalar")


def test_called_status_move_is_not_av_evidence():
    # a [from]-called move was never SELECTED, so the vest survives
    o = _obs(["|switch|p2a: Komala|Komala, M|100/100",
              "|move|p2a: Komala|Yawn|p1a: X|[from]move: Sleep Talk",
              "|turn|2"])
    assert "assaultvest" not in o.forbidden("komala")


def test_damaging_move_is_not_av_evidence():
    o = _obs(["|switch|p2a: Slowking|Slowking-Galar, M|100/100",
              "|move|p2a: Slowking|Sludge Bomb|p1a: X",
              "|turn|2"])
    assert "assaultvest" not in o.forbidden("slowkinggalar")


# ---- Regenerator from re-switch HP gain --------------------------------

REGEN_STORY = [
    "|switch|p2a: Slowking|Slowking-Galar, M|100/100",
    "|move|p1a: Us|Knock Off|p2a: Slowking",
    "|-damage|p2a: Slowking|60/100",
    "|switch|p2a: Gholdengo|Gholdengo|100/100",
    "|turn|3",
    "|switch|p2a: Slowking|Slowking-Galar, M|93/100",   # +33 on the bench
    "|turn|4",
]


def test_bench_heal_on_reswitch_proves_regenerator():
    o = _obs(REGEN_STORY)
    assert "slowkinggalar" in o.regen


def test_no_gain_is_no_regen_evidence():
    story = [e.replace("93/100", "60/100") for e in REGEN_STORY]
    assert "slowkinggalar" not in _obs(story).regen


def test_regen_incapable_species_never_flagged():
    # same HP story on a species whose dex has no Regenerator
    story = [e.replace("Slowking-Galar, M", "Gholdengo")
              .replace("p2a: Slowking", "p2a: Ghol")
              .replace("Gholdengo|Gholdengo", "Kingambit|Kingambit")
             for e in REGEN_STORY]
    o = _obs(story)
    assert "gholdengo" not in o.regen


def test_revival_blessing_return_is_not_regen():
    o = _obs(["|switch|p2a: Slowking|Slowking-Galar, M|100/100",
              "|-damage|p2a: Slowking|40/100",
              "|faint|p2a: Slowking",
              "|switch|p2a: Gholdengo|Gholdengo|100/100",
              "|turn|3",
              "|switch|p2a: Slowking|Slowking-Galar, M|50/100",  # revived
              "|turn|4"])
    assert "slowkinggalar" not in o.regen


# ---- negative ability evidence from a silent first entry ---------------

def test_silent_landorus_rules_out_intimidate():
    o = _obs(["|switch|p2a: Landorus|Landorus-Therian, M|100/100",
              "|turn|2"])
    assert "intimidate" in o.ability_forbidden("landorustherian")


def test_announced_intimidate_stays_possible():
    o = _obs(["|switch|p2a: Landorus|Landorus-Therian, M|100/100",
              "|-ability|p2a: Landorus|Intimidate|boost",
              "|-unboost|p1a: Us|atk|1",
              "|turn|2"])
    assert "intimidate" not in o.ability_forbidden("landorustherian")


def test_silent_torkoal_rules_out_drought_only_without_sun():
    silent = _obs(["|switch|p2a: Torkoal|Torkoal, M|100/100", "|turn|2"])
    assert "drought" in silent.ability_forbidden("torkoal")
    # sun already up: Drought is silent legitimately — no evidence
    sunny = _obs(["|-weather|SunnyDay",
                  "|switch|p2a: Torkoal|Torkoal, M|100/100", "|turn|2"])
    assert "drought" not in sunny.ability_forbidden("torkoal")


def test_weather_announce_via_from_tag_counts():
    o = _obs(["|switch|p2a: Torkoal|Torkoal, M|100/100",
              "|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Torkoal",
              "|turn|2"])
    assert "drought" not in o.ability_forbidden("torkoal")


def test_terrain_setter_silent_without_terrain_is_ruled_out():
    o = _obs(["|switch|p2a: Rillaboom|Rillaboom, M|100/100", "|turn|2"])
    assert "grassysurge" in o.ability_forbidden("rillaboom")


def test_second_entry_is_never_evidence():
    # Dauntless Shield announces once per battle: only the FIRST entry
    # opens the watch, so a silent re-entry proves nothing
    o = _obs(["|switch|p2a: Zamazenta|Zamazenta|100/100",
              "|-ability|p2a: Zamazenta|Dauntless Shield|boost",
              "|switch|p2a: Gholdengo|Gholdengo|100/100",
              "|turn|2",
              "|switch|p2a: Zamazenta|Zamazenta|100/100",
              "|turn|3"])
    assert "dauntlessshield" not in o.ability_forbidden("zamazenta")


def test_neutralizing_gas_disables_the_rule():
    o = _obs(["|switch|p2a: Weezing|Weezing-Galar, F|100/100",
              "|-ability|p2a: Weezing|Neutralizing Gas",
              "|switch|p2a: Landorus|Landorus-Therian, M|100/100",
              "|turn|2"])
    assert "intimidate" not in o.ability_forbidden("landorustherian")


def test_own_side_entries_are_not_latched():
    o = _obs(["|switch|p1a: Landorus|Landorus-Therian, M|100/100",
              "|turn|2"])
    assert o.ability_forbidden("landorustherian") == frozenset()


# ---- ident-vs-details fixes the trio exposed ---------------------------

def test_forme_upkeep_proof_uses_details_species():
    """The ident nickname is the BASE name for formes ('p2a: Slowking' for
    Slowking-Galar); keyed by ident, every upkeep proof missed forme mons —
    the Leftovers elimination must land on the details key consumers query."""
    o = _obs(["|switch|p2a: Slowking|Slowking-Galar, M|100/100",
              "|-damage|p2a: Slowking|60/100",
              "|turn|2"])
    assert "leftovers" in o.forbidden("slowkinggalar")


def test_par_tracking_is_alive():
    """The par branches sat AFTER a broader -status elif in the same chain:
    dead code, _par stayed empty, the speed-order par corrections never
    fired. Now merged into the live branch, keyed by details species."""
    o = _obs(["|switch|p2a: Landorus|Landorus-Therian, M|100/100",
              "|-status|p2a: Landorus|par",
              "|turn|2"])
    assert "p2 landorustherian" in o._par


# ---- on-hit negative ability evidence (2026-08-05, user idea) ----------

HIT = ["|switch|p1a: Ninetales|Ninetales, F|100/100",
       "|switch|p2a: Clodsire|Clodsire, M|100/100"]


def test_water_damage_rules_out_the_absorbers():
    """The Clodsire case: one Water hit that connects proves
    not-waterabsorb — which in practice means Unaware."""
    o = _obs(HIT + ["|move|p1a: Ninetales|Surf|p2a: Clodsire",
                    "|-damage|p2a: Clodsire|70/100", "|turn|2"])
    assert "waterabsorb" in o.ability_forbidden("clodsire")
    assert "stormdrain" in o.ability_forbidden("clodsire")


def test_immune_announce_means_no_damage_and_no_forbid():
    o = _obs(HIT + ["|move|p1a: Ninetales|Surf|p2a: Clodsire",
                    "|-immune|p2a: Clodsire|[from] ability: Water Absorb",
                    "|turn|2"])
    assert "waterabsorb" not in o.ability_forbidden("clodsire")
    assert o.revealed_ability.get("clodsire") == "waterabsorb"


def test_from_tagged_damage_is_not_hit_evidence():
    # hazard chip while our attack is armed must not convict
    o = _obs(HIT + ["|move|p1a: Ninetales|Surf|p2a: Clodsire",
                    "|-damage|p2a: Clodsire|94/100|[from] Stealth Rock",
                    "|turn|2"])
    assert "waterabsorb" not in o.ability_forbidden("clodsire")


def test_their_self_cost_damage_is_not_hit_evidence():
    # their Substitute cost is a bare -damage right after THEIR move
    o = _obs(HIT + ["|move|p1a: Ninetales|Surf|p2a: Clodsire",
                    "|move|p2a: Clodsire|Substitute|p2a: Clodsire",
                    "|-start|p2a: Clodsire|Substitute",
                    "|-damage|p2a: Clodsire|75/100", "|turn|2"])
    assert "waterabsorb" not in o.ability_forbidden("clodsire")


def test_sound_flag_and_priority_reactors():
    o = _obs(HIT + ["|move|p1a: Ninetales|Hyper Voice|p2a: Clodsire",
                    "|-damage|p2a: Clodsire|80/100", "|turn|2"])
    assert "soundproof" in o.ability_forbidden("clodsire")
    o2 = _obs(HIT + ["|move|p1a: Ninetales|Quick Attack|p2a: Clodsire",
                     "|-damage|p2a: Clodsire|90/100", "|turn|2"])
    assert "dazzling" in o2.ability_forbidden("clodsire")
    assert "armortail" in o2.ability_forbidden("clodsire")


def test_mold_breaker_attacker_voids_the_evidence():
    events = ["|switch|p1a: Excadrill|Excadrill, M|100/100",
              "|switch|p2a: Clodsire|Clodsire, M|100/100",
              "|move|p1a: Excadrill|Earthquake|p2a: Clodsire",
              "|-damage|p2a: Clodsire|60/100", "|turn|2"]
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role="p1",
        team={"p1: Excadrill": SimpleNamespace(species="Excadrill",
                                               ability="moldbreaker")})
    o = BattleObservations()
    o.update(b)
    assert "levitate" not in o.ability_forbidden("clodsire")
