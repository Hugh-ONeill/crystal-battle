"""Direct item/ability reveals from protocol [from]/[of] tags.

Not inference: the sim broadcasts the source ("Gliscor was poisoned by
Toxic Orb" = |-status|...|tox|[from] item: Toxic Orb) and the scanner was
dropping the tag — the audited game had item: null at T16 with the orb
announced since T2. Same family: Flame Orb on Guts users, Leftovers,
Rocky Helmet chip ([of] flips the owner), ability activations."""
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


def test_toxic_orb_status_reveals_item():
    o = _obs(["|switch|p2a: Gliscor|Gliscor, M|100/100",
              "|-status|p2a: Gliscor|tox|[from] item: Toxic Orb",
              "|turn|2"])
    assert o.revealed_item.get("gliscor") == "toxicorb"


def test_flame_orb_burn_reveals_item():
    # the Guts-user case: burned off the bat = orb announced end of turn 1
    o = _obs(["|switch|p2a: Ursaluna|Ursaluna, F|100/100",
              "|-status|p2a: Ursaluna|brn|[from] item: Flame Orb",
              "|turn|2"])
    assert o.revealed_item.get("ursaluna") == "flameorb"


def test_poison_heal_heal_reveals_ability():
    o = _obs(["|switch|p2a: Gliscor|Gliscor, M|100/100",
              "|-status|p2a: Gliscor|tox|[from] item: Toxic Orb",
              "|-heal|p2a: Gliscor|100/100 tox|[from] ability: Poison Heal",
              "|turn|3"])
    assert o.revealed_ability.get("gliscor") == "poisonheal"
    assert o.revealed_item.get("gliscor") == "toxicorb"


def test_leftovers_heal_reveals_item():
    o = _obs(["|switch|p2a: Ting-Lu|Ting-Lu|100/100",
              "|-heal|p2a: Ting-Lu|100/100|[from] item: Leftovers",
              "|turn|2"])
    assert o.revealed_item.get("tinglu") == "leftovers"


def test_rocky_helmet_of_tag_flips_owner():
    # OUR mon takes the chip; the helmet belongs to the [of] mon (theirs)
    o = _obs(["|switch|p1a: Cinderace|Cinderace, M|100/100",
              "|switch|p2a: Garchomp|Garchomp, F|100/100",
              "|-damage|p1a: Cinderace|84/100|[from] item: Rocky Helmet"
              "|[of] p2a: Garchomp",
              "|turn|2"])
    assert o.revealed_item.get("garchomp") == "rockyhelmet"
    assert "cinderace" not in o.revealed_item


def test_our_side_reveals_are_not_recorded():
    o = _obs(["|switch|p1a: Gliscor|Gliscor, M|100/100",
              "|-status|p1a: Gliscor|tox|[from] item: Toxic Orb",
              "|turn|2"])
    assert o.revealed_item == {}


def test_static_paralysis_of_tag_reveals_ability():
    o = _obs(["|switch|p1a: Cinderace|Cinderace, M|100/100",
              "|switch|p2a: Zapdos|Zapdos|100/100",
              "|-status|p1a: Cinderace|par|[from] ability: Static"
              "|[of] p2a: Zapdos",
              "|turn|2"])
    assert o.revealed_ability.get("zapdos") == "static"


def test_no_tags_no_reveals():
    o = _obs(["|switch|p2a: Gliscor|Gliscor, M|100/100",
              "|-status|p2a: Gliscor|tox",
              "|turn|2"])
    assert o.revealed_item == {} and o.revealed_ability == {}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
