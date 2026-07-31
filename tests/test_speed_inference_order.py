# Speed-floor inference must try the CHEAPEST explanation first.
#
# Caught live on a commentary demo 2026-07-31: our Iron Crown was paralysed by
# Thunder Wave (paralysis halves Speed), a Slowking-Galar moved first, and the
# engine announced a Choice Scarf. But the floor a paralysed Iron Crown sets
# (~153) is cleared by a fully Speed-invested Slowking-Galar (~174) holding
# nothing at all — the observation admitted a no-item explanation and the
# engine picked the exotic one.
#
# The old order was: floor exceeded -> assign Choice Scarf -> only then check
# whether max Speed was ALSO needed. "They invested more Speed than my
# canonical bulky spread assumed" is the commoner hypothesis and must be tested
# first. This is not cosmetic: a believed Choice Scarf also means a believed
# CHOICE-LOCK, so the search plans against one move the target can freely
# switch off.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.gen9_translator import _calc_stat_modern
from showdown.set_inference import BattleObservations

# Slowking-Galar base Spe 30, Iron Crown base Spe 90
SG_CANON = _calc_stat_modern(30, 31, 0, 100, 1.0, False)    # bulky, 0 EV
SG_MAX = _calc_stat_modern(30, 31, 252, 100, 1.1, False)    # 252 EV, +nature
IC_MAX = _calc_stat_modern(90, 31, 252, 100, 1.1, False)


def test_investment_explains_a_paralysis_lowered_floor():
    """The live case: max investment clears the floor, so no item is implied."""
    obs = BattleObservations()
    obs.speed_floor["slowkinggalar"] = IC_MAX * 0.5      # paralysed Iron Crown
    # the canonical bulky spread genuinely cannot explain being outsped
    assert obs.scarf_needed("slowkinggalar", SG_CANON, "leftovers")
    # ...but full investment can, with no item at all
    assert obs.max_speed_suffices("slowkinggalar", SG_MAX)


def test_scarf_still_inferred_when_investment_cannot_reach():
    """A floor beyond any legal spread must still promote the item — the fix
    must not make the engine blind to real Scarfs."""
    obs = BattleObservations()
    obs.speed_floor["slowkinggalar"] = SG_MAX * 1.4      # unreachable unaided
    assert obs.scarf_needed("slowkinggalar", SG_CANON, "leftovers")
    assert not obs.max_speed_suffices("slowkinggalar", SG_MAX)


def test_no_floor_means_no_claim():
    obs = BattleObservations()
    assert not obs.scarf_needed("slowkinggalar", SG_CANON, "leftovers")
    assert not obs.max_speed_suffices("slowkinggalar", SG_MAX)


def test_our_paralysis_lowers_the_floor_it_sets():
    """Guards the correction the floor logic already makes: a paralysed mon of
    ours proves far less about the opponent's Speed than a healthy one."""
    healthy, paralysed = IC_MAX, IC_MAX * 0.5
    obs = BattleObservations()
    obs.speed_floor["slowkinggalar"] = paralysed
    assert obs.max_speed_suffices("slowkinggalar", SG_MAX)
    obs.speed_floor["slowkinggalar"] = healthy
    assert not obs.max_speed_suffices("slowkinggalar", SG_MAX)
