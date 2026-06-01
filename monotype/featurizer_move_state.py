"""
Per-turn state-context featurizer for the move-prediction net.

Builds on top of the lead-preview featurizer (which only encodes static
mon info) by adding the *dynamic* state that drives move choice:

  - active mon's current boost stages (atk/def/spa/spd/spe each in [-6, +6])
  - active mon's status (none/brn/par/slp/frz/psn/tox)
  - field: weather (none/sun/rain/snow/sand/harshsun/heavyrain), terrain
    (none/electric/grassy/psychic/misty), trick room, tailwind per side
  - hazards on both sides: SR / spikes count / toxic spikes count / web
  - screens on both sides: reflect / light screen / aurora veil

Encoding is **actor-relative**: features are laid out as
(actor_own, actor_opp) so the net learns symmetric patterns regardless of
which player the actor is.

Output: (STATE_DIM,) numpy float32 vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Canonical orderings (so featurization is stable across runs).
BOOST_STATS = ("atk", "def", "spa", "spd", "spe")  # 5 per side
STATUSES = ("none", "brn", "par", "slp", "frz", "psn", "tox")  # 7 one-hot per side
WEATHERS = ("none", "sun", "rain", "snow", "sand", "harshsun", "heavyrain")  # 7
TERRAINS = ("none", "electric", "grassy", "psychic", "misty")  # 5


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


@dataclass
class TurnState:
    """Per-turn dynamic state — populated by the replay-log walker."""
    # Per-side active state
    boosts_p1: dict = field(default_factory=lambda: dict.fromkeys(BOOST_STATS, 0))
    boosts_p2: dict = field(default_factory=lambda: dict.fromkeys(BOOST_STATS, 0))
    status_p1: str = "none"
    status_p2: str = "none"
    # Field state
    weather: str = "none"
    terrain: str = "none"
    trick_room: bool = False
    tailwind_p1: bool = False
    tailwind_p2: bool = False
    # Hazards (cumulative — spikes/toxicspikes can stack)
    sr_p1: bool = False
    spikes_p1: int = 0       # 0-3
    tspikes_p1: int = 0      # 0-2
    web_p1: bool = False
    sr_p2: bool = False
    spikes_p2: int = 0
    tspikes_p2: int = 0
    web_p2: bool = False
    # Screens
    reflect_p1: bool = False
    lightscreen_p1: bool = False
    auroraveil_p1: bool = False
    reflect_p2: bool = False
    lightscreen_p2: bool = False
    auroraveil_p2: bool = False

    def reset_active_boosts(self, side: str):
        """Switch wipes the outgoing mon's boosts (poke-engine convention)."""
        target = self.boosts_p1 if side == "p1" else self.boosts_p2
        for k in target:
            target[k] = 0

    def reset_active_status(self, side: str):
        """Boots-style switch-in: status carries over (unlike boosts). Only
        Natural Cure / Regenerator-style mechanics clear status on switch;
        the log will emit |-curestatus| if that happens. So this method
        exists only for explicit faint-switch transitions where the new
        mon has its own (initially none) status. The extractor uses it to
        reset on switch-in."""
        if side == "p1":
            self.status_p1 = "none"
        else:
            self.status_p2 = "none"


# Dimension layout, actor-relative:
#   [0:5]   own boost stages (atk/def/spa/spd/spe), each in [-6, 6] / 6
#   [5:10]  opp boost stages
#   [10:17] own status one-hot
#   [17:24] opp status one-hot
#   [24:31] weather one-hot
#   [31:36] terrain one-hot
#   [36]    trick_room
#   [37]    own tailwind
#   [38]    opp tailwind
#   [39:43] own hazards (sr, spikes/3, tspikes/2, web)
#   [43:47] opp hazards
#   [47:50] own screens (reflect, ls, aurora)
#   [50:53] opp screens
STATE_DIM = 53


def featurize_turn_state(ts: TurnState, actor: str) -> np.ndarray:
    """Return (STATE_DIM,) features, actor-relative."""
    if actor == "p1":
        own_boosts, opp_boosts = ts.boosts_p1, ts.boosts_p2
        own_status, opp_status = ts.status_p1, ts.status_p2
        own_tw, opp_tw = ts.tailwind_p1, ts.tailwind_p2
        own_sr, opp_sr = ts.sr_p1, ts.sr_p2
        own_sp, opp_sp = ts.spikes_p1, ts.spikes_p2
        own_ts, opp_ts = ts.tspikes_p1, ts.tspikes_p2
        own_web, opp_web = ts.web_p1, ts.web_p2
        own_re, opp_re = ts.reflect_p1, ts.reflect_p2
        own_ls, opp_ls = ts.lightscreen_p1, ts.lightscreen_p2
        own_av, opp_av = ts.auroraveil_p1, ts.auroraveil_p2
    else:
        own_boosts, opp_boosts = ts.boosts_p2, ts.boosts_p1
        own_status, opp_status = ts.status_p2, ts.status_p1
        own_tw, opp_tw = ts.tailwind_p2, ts.tailwind_p1
        own_sr, opp_sr = ts.sr_p2, ts.sr_p1
        own_sp, opp_sp = ts.spikes_p2, ts.spikes_p1
        own_ts, opp_ts = ts.tspikes_p2, ts.tspikes_p1
        own_web, opp_web = ts.web_p2, ts.web_p1
        own_re, opp_re = ts.reflect_p2, ts.reflect_p1
        own_ls, opp_ls = ts.lightscreen_p2, ts.lightscreen_p1
        own_av, opp_av = ts.auroraveil_p2, ts.auroraveil_p1

    out = np.zeros(STATE_DIM, dtype=np.float32)
    for i, st in enumerate(BOOST_STATS):
        out[i] = own_boosts[st] / 6.0
        out[5 + i] = opp_boosts[st] / 6.0

    status_idx = {s: i for i, s in enumerate(STATUSES)}
    out[10:17] = _onehot(status_idx.get(own_status, 0), 7)
    out[17:24] = _onehot(status_idx.get(opp_status, 0), 7)

    weather_idx = {w: i for i, w in enumerate(WEATHERS)}
    out[24:31] = _onehot(weather_idx.get(ts.weather, 0), 7)

    terrain_idx = {t: i for i, t in enumerate(TERRAINS)}
    out[31:36] = _onehot(terrain_idx.get(ts.terrain, 0), 5)

    out[36] = float(ts.trick_room)
    out[37] = float(own_tw)
    out[38] = float(opp_tw)

    out[39] = float(own_sr)
    out[40] = own_sp / 3.0
    out[41] = own_ts / 2.0
    out[42] = float(own_web)
    out[43] = float(opp_sr)
    out[44] = opp_sp / 3.0
    out[45] = opp_ts / 2.0
    out[46] = float(opp_web)

    out[47] = float(own_re)
    out[48] = float(own_ls)
    out[49] = float(own_av)
    out[50] = float(opp_re)
    out[51] = float(opp_ls)
    out[52] = float(opp_av)

    return out


# Move-name normalizations for parsing -fieldstart / -sidestart lines.
def normalize_weather(name: str) -> str:
    n = name.lower().replace(" ", "").replace("-", "")
    return {
        "sunnyday": "sun", "raindance": "rain", "snow": "snow",
        "snowscape": "snow", "snowyday": "snow", "sandstorm": "sand",
        "desolateland": "harshsun", "primordialsea": "heavyrain",
    }.get(n, "none")


def normalize_terrain(name: str) -> str:
    n = name.lower().replace(" ", "").replace("-", "")
    return {
        "electricterrain": "electric", "grassyterrain": "grassy",
        "psychicterrain": "psychic", "mistyterrain": "misty",
    }.get(n, "none")


# Showdown status code map (lowercase): brn/par/slp/frz/psn/tox
def normalize_status(code: str) -> str:
    c = code.lower().strip()
    return c if c in STATUSES else "none"
