"""Set-coherence rules: the item axis must respect revealed ground truth.

The audited specimen (game 14287138 T16, worry-miner catch 2026-08-02):
both worlds assumed Choice Band + Poison Heal Gliscor after all four stall
moves (EQ/Protect/Toxic/Knock Off) were revealed — chaos marginals compose
per-axis, so the item draw never saw the moves. These tests pin the rules:
revealed status/boost moves veto Choice items and Assault Vest (except the
real Choice status techs), Poison Heal forces Toxic Orb, and known
item/ability always win over any rule.
"""

import random

import pytest

from showdown import chaos_stats
from showdown.chaos_stats import PokemonStats, incompatible_items


def mk_stats(items, abilities=None, moves=None):
    return PokemonStats("Testmon", {
        "Raw count": 100,
        "usage": 0.5,
        "Moves": moves or {"earthquake": 100, "protect": 100,
                           "toxic": 100, "knockoff": 100},
        "Items": items,
        "Abilities": abilities or {"poisonheal": 100},
        "Spreads": {"Careful:244/0/36/0/228/0": 100},
        "Tera Types": {"water": 100},
    })


GLISCOR_REVEALS = ("earthquake", "protect", "toxic", "knockoff")


def test_status_reveal_vetoes_choice_and_vest():
    assert incompatible_items(("protect",)) == frozenset(
        {"assaultvest", "choiceband", "choicespecs", "choicescarf"})


def test_boost_reveal_vetoes_choice():
    # user rule 2026-08-02: an observed boost move rules Choice items out
    assert "choiceband" in incompatible_items(("swordsdance",))
    assert "choicescarf" in incompatible_items(("bulkup", "earthquake"))


def test_choice_status_techs_only_veto_vest():
    for tech in ("trick", "switcheroo", "healingwish", "lunardance",
                 "memento"):
        assert incompatible_items((tech,)) == frozenset({"assaultvest"})


def test_damaging_reveals_veto_nothing():
    assert incompatible_items(("rapidspin", "headlongrush")) == frozenset()
    assert incompatible_items(()) == frozenset()


def test_gliscor_specimen_sampled():
    """The audited case: CB can never survive four revealed stall moves."""
    st = mk_stats({"choiceband": 50, "toxicorb": 50},
                  {"poisonheal": 99, "hypercutter": 1})
    rng = random.Random(1)
    for _ in range(50):
        s = st.sample_set(rng, known_moves=GLISCOR_REVEALS)
        assert s["item"] == "toxicorb"


def test_poison_heal_forces_toxic_orb_without_reveals():
    st = mk_stats({"choiceband": 90, "toxicorb": 10})
    rng = random.Random(2)
    for _ in range(20):
        assert st.sample_set(rng)["item"] == "toxicorb"


def test_known_item_beats_every_rule():
    st = mk_stats({"toxicorb": 100})
    s = st.sample_set(random.Random(3), known_moves=("protect",),
                      known_item="choiceband")
    assert s["item"] == "choiceband"


def test_known_ability_is_kept_and_conditions_item():
    st = mk_stats({"choiceband": 90, "toxicorb": 10},
                  {"hypercutter": 90, "poisonheal": 10})
    s = st.sample_set(random.Random(4), known_moves=(),
                      known_ability="poisonheal")
    assert s["ability"] == "poisonheal"
    assert s["item"] == "toxicorb"


def test_speed_pessimistic_scarf_respects_reveals():
    st = mk_stats({"choicescarf": 50, "leftovers": 50},
                  {"hypercutter": 100})
    rng = random.Random(5)
    s = st.sample_set(rng, known_moves=("protect",), speed_pessimistic=True)
    assert s["item"] == "leftovers"
    # without a contradicting reveal the pessimistic scarf stays
    s = st.sample_set(rng, speed_pessimistic=True)
    assert s["item"] == "choicescarf"


def test_deterministic_top_item_exclude():
    st = mk_stats({"choiceband": 90, "leftovers": 10})
    assert st.top_item() == "choiceband"
    assert st.top_item(exclude=incompatible_items(("toxic",))) == "leftovers"


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(chaos_stats, "_COHERENCE_ON", False)
    assert incompatible_items(("protect", "swordsdance")) == frozenset()
    st = mk_stats({"choiceband": 100}, {"poisonheal": 100})
    s = st.sample_set(random.Random(6), known_moves=GLISCOR_REVEALS)
    assert s["item"] == "choiceband"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
