"""The overlay's resource gate must read the PROSE `resource` field.

Introduced 2026-08-02 and caught the same day by the overlay's own worry
channel: the gate compared `resource` to a weather name by exact string
equality, so it matched only the five single-token values ("sun", "rain",
"sand", "snow", "grassyterrain"). Every entry written with prose — "Psychic
Terrain", "Trick Room turns", "snow + Aurora Veil", "Tailwind turns (on the
Prankster builds)" — could never match, so `weather-down:<species>` fired on
EVERY turn that mon was alive: 5 of 11 resource holders permanently
false-positive. A resource can also live in three different places (weather,
fields, our side conditions) and the old gate only looked in one.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.overlay import OverlayShadow


@dataclass(frozen=True)
class Ev:
    name: str


@pytest.fixture(scope="module")
def sh():
    o = OverlayShadow.__new__(OverlayShadow)
    o.roles = json.loads(
        (Path(__file__).parent.parent / "showdown" / "roles.json").read_text()
    )["roles"]
    o._setup_moves = set()
    return o


def reasons(sh, species, weather=None, fields=None, side=None):
    team = {"a": NS(species=species, fainted=False, active=False,
                    current_hp_fraction=1.0),
            "b": NS(species="Kingambit", fainted=False, active=True,
                    current_hp_fraction=1.0)}
    b = NS(team=team, weather=weather or {}, fields=fields or {},
           side_conditions=side or {})
    return sh._role_reasons(b, [NS(move_choice="tackle", visits=10)])


@pytest.mark.parametrize("species,up_kwargs", [
    ("Ninetales", {"weather": {Ev("SUNNYDAY"): 1}}),          # weather
    ("Indeedee", {"fields": {Ev("PSYCHIC_TERRAIN"): 1}}),     # terrain field
    ("Cresselia", {"fields": {Ev("TRICK_ROOM"): 1}}),         # trick room
    ("Whimsicott", {"side": {Ev("TAILWIND"): 1}}),            # side condition
])
def test_gate_silent_when_the_resource_is_up(sh, species, up_kwargs):
    assert not [r for r in reasons(sh, species, **up_kwargs)
                if r.startswith("weather-down")]


@pytest.mark.parametrize("species", ["Ninetales", "Indeedee", "Cresselia"])
def test_gate_fires_when_the_resource_is_down(sh, species):
    assert any(r.startswith("weather-down") for r in reasons(sh, species))


def test_prose_resource_values_are_all_reachable(sh):
    """Every `resource` in the file must normalise to at least one token, or
    its gate can never be satisfied."""
    from showdown.team_roles import resource_tokens
    for species, e in sh.roles.items():
        res = e.get("resource")
        if res:
            assert resource_tokens(res), f"{species}: {res!r} yields no token"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
