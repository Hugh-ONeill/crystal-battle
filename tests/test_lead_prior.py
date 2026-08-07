"""Preview lead prior: nudges the lead ranking with MEASURED per-species
deltas (showdown/lead_measured.json). Policy-shaped on purpose — it steers
one decision rather than scoring a state.

The guard these tests exist for: only entries whose 95% CI EXCLUDES ZERO may
enter the table. On 2026-08-07 a Ting-Lu "pilot gap" was flagged off a point
estimate whose interval spanned zero, and 12 of 17 measured species are
indistinguishable from zero — so a table built on point estimates would
encode mostly noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_player import _apply_lead_prior, _lead_prior_table

SPECIES = ["Glimmora", "Iron Moth", "Ditto", "Ceruledge", "Dragapult",
           "Kingambit"]


def test_table_admits_only_significant_entries():
    t = _lead_prior_table()
    assert "kingambit" in t and t["kingambit"] < 0      # -8.9 [-16.4,-1.3]
    assert "glimmora" in t and t["glimmora"] > 0        # +12.5 [+6.6,+18.3]
    # spans zero at our LARGEST sample (1062 games) — must not be encoded
    assert "tinglu" not in t
    assert "dragapult" not in t                         # +3.1 [-1.4,+7.7]


def test_prior_demotes_kingambit_from_the_argmax():
    vals = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53]         # Kingambit is #1
    assert SPECIES[max(range(6), key=lambda i: vals[i])] == "Kingambit"
    out, applied = _apply_lead_prior(vals, SPECIES, 1.0)
    assert SPECIES[max(range(6), key=lambda i: out[i])] == "Glimmora"
    assert any(a.startswith("kingambit-") for a in applied)


def test_scale_zero_and_unknown_species_are_no_ops():
    vals = [0.5] * 6
    out, applied = _apply_lead_prior(vals, SPECIES, 0.0)
    assert out == vals and applied == []
    out, applied = _apply_lead_prior([0.5, 0.5], ["Blissey", "Skarmory"], 1.0)
    assert out == [0.5, 0.5] and applied == []


def test_scale_is_linear():
    vals = [0.5] * 6
    half, _ = _apply_lead_prior(vals, SPECIES, 0.5)
    full, _ = _apply_lead_prior(vals, SPECIES, 1.0)
    kg = SPECIES.index("Kingambit")
    assert abs((full[kg] - 0.5) - 2 * (half[kg] - 0.5)) < 1e-9


# ---- opening record (2026-08-07): who they led, did we repudiate ours ----
from types import SimpleNamespace                                   # noqa: E402

from showdown.set_inference import BattleObservations               # noqa: E402


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


def test_opening_record_captures_their_lead_and_our_t1_switch():
    o = _obs(["|switch|p1a: Glimmora|Glimmora, M|100/100",
              "|switch|p2a: Kingambit|Kingambit, M|100/100",
              "|turn|1",
              "|switch|p1a: Dragapult|Dragapult, M|100/100",
              "|turn|2"])
    assert o.opp_lead == "kingambit"
    assert o.our_t1_switch is True


def test_a_later_switch_is_not_t1_churn():
    """The window is turn 1 only — a turn-2 switch is ordinary play, and
    counting it would inflate the churn rate the lead analysis depends on."""
    o = _obs(["|switch|p1a: Glimmora|Glimmora, M|100/100",
              "|switch|p2a: Kingambit|Kingambit, M|100/100",
              "|turn|1",
              "|move|p1a: Glimmora|Stealth Rock|p2a: Kingambit",
              "|turn|2",
              "|switch|p1a: Dragapult|Dragapult, M|100/100"])
    assert o.opp_lead == "kingambit"
    assert o.our_t1_switch is False


# ---- the knob's semantics: calibrated strength, still default OFF --------
import os                                                           # noqa: E402

from showdown.gen9_player import (LEAD_PRIOR_SCALE,                 # noqa: E402
                                  _lead_prior_scale)


def test_knob_is_off_unless_asked_and_on_means_the_calibrated_scale(
        monkeypatch):
    """0.25 is the MEASURED nudge strength (x1.0 was an override that zeroed
    two species and manufactured turn-1 churn). Enabling must not require
    remembering the number, and must still be a deliberate act."""
    monkeypatch.delenv("CB_LEAD_PRIOR", raising=False)
    assert _lead_prior_scale() == 0.0            # default OFF
    for off in ("0", "off", "false", ""):
        monkeypatch.setenv("CB_LEAD_PRIOR", off)
        assert _lead_prior_scale() == 0.0
    for on in ("1", "on", "true"):
        monkeypatch.setenv("CB_LEAD_PRIOR", on)
        assert _lead_prior_scale() == LEAD_PRIOR_SCALE == 0.25
    monkeypatch.setenv("CB_LEAD_PRIOR", "0.5")   # explicit scale for sweeps
    assert _lead_prior_scale() == 0.5
    monkeypatch.setenv("CB_LEAD_PRIOR", "garbage")
    assert _lead_prior_scale() == 0.0            # never crash a live game
