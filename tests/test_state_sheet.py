"""Board fact sheet: determinism, fixed slots, and the revealed/assumed
split. The renderer feeds a live consult path, so the contract under test
is as much "never lies, never breaks" as any particular formatting: a
sampled item printed as a reveal is the exact laundering the sheet exists
to prevent, and an exception here would kill the consult thread.
"""

from types import SimpleNamespace

import poke_engine as pe

from showdown.state_sheet import render_sheet


# ---- builders ----------------------------------------------------------

def mv(mid, pp=16, disabled=False):
    return pe.Move(id=mid, pp=pp, disabled=disabled)


def mon(mid, hp=100, maxhp=100, item="none", ability="none",
        moves=(), status="none", tera="normal", terastallized=False,
        nature="serious", evs=(85,) * 6, speed=100, rest=0, slept=0):
    return pe.Pokemon(
        id=mid, hp=hp, maxhp=maxhp, item=item, ability=ability,
        moves=[mv(m) if isinstance(m, str) else m for m in moves],
        status=status, tera_type=tera, terastallized=terastallized,
        nature=nature, evs=tuple(evs), speed=speed,
        rest_turns=rest, sleep_turns=slept)


def side(mons, **kw):
    return pe.Side(pokemon=mons, **kw)


def state(s1, s2, **kw):
    return pe.State(side_one=s1, side_two=s2, **kw)


def bmon(species, moves=(), item=None, ability=None, tera=None,
         fainted=False):
    return SimpleNamespace(species=species, fainted=fainted,
                           moves={m: None for m in moves},
                           item=item, ability=ability, tera_type=tera)


def battle(turn=12, team=(), opp=()):
    return SimpleNamespace(turn=turn,
                           team={m.species: m for m in team},
                           opponent_team={m.species: m for m in opp})


def default_state():
    ours = [mon("kingambit", hp=178, maxhp=244, item="leftovers",
                ability="supremeoverlord", tera="dark",
                moves=["suckerpunch", "swordsdance", "ironhead",
                       "kowtowcleave"]),
            mon("gliscor", item="toxicorb", ability="poisonheal",
                tera="water", moves=["earthquake", "protect"])]
    theirs = [mon("greattusk", hp=64, maxhp=100, item="heavydutyboots",
                  ability="protosynthesis", tera="steel", nature="jolly",
                  evs=(0, 252, 0, 0, 4, 252), speed=290,
                  moves=["headlongrush", "rapidspin", "closecombat",
                         "icespinner"]),
              mon("gholdengo", item="airballoon", ability="goodasgold",
                  tera="fairy", moves=["makeitrain", "shadowball"])]
    return state(side(ours), side(theirs))


# ---- contract ----------------------------------------------------------

def test_deterministic():
    b = battle(opp=[bmon("greattusk", moves=["headlongrush"])])
    assert (render_sheet(default_state(), b)
            == render_sheet(default_state(), b))


def test_never_raises_on_garbage():
    assert isinstance(render_sheet(object()), str)
    assert isinstance(render_sheet(None), str)
    assert "unavailable" not in render_sheet(default_state())


def test_fixed_slots_print_explicit_absence():
    sheet = render_sheet(default_state(), battle())
    for needle in ("weather=none", "terrain=none", "trickroom=no",
                   "US hazards: SR=no spikes=0 tspikes=0 web=no",
                   "THEM hazards: SR=no",
                   "screens: none", "boosts: none", "volatiles: none",
                   "fainted: none"):
        assert needle in sheet, needle


# ---- revealed vs assumed -----------------------------------------------

def test_revealed_and_assumed_moves_split():
    b = battle(opp=[bmon("greattusk", moves=["headlongrush", "rapidspin"])])
    sheet = render_sheet(default_state(), b)
    line_rev = next(l for l in sheet.splitlines() if "revealed:" in l)
    line_asm = next(l for l in sheet.splitlines() if "assumed" in l
                    and "closecombat" in l)
    assert "headlongrush" in line_rev and "rapidspin" in line_rev
    assert "closecombat" in line_asm and "icespinner" in line_asm
    assert "closecombat" not in line_rev


def test_unrevealed_item_is_assumed_revealed_item_is_fact():
    hidden = battle(opp=[bmon("greattusk")])
    sheet = render_sheet(default_state(), hidden)
    asm = next(l for l in sheet.splitlines()
               if "assumed" in l and "heavydutyboots" in l)
    assert "item=heavydutyboots" in asm
    assert "revealed: nothing yet" in sheet

    shown = battle(opp=[bmon("greattusk", item="heavydutyboots")])
    sheet2 = render_sheet(default_state(), shown)
    rev = next(l for l in sheet2.splitlines() if "revealed:" in l
               and "heavydutyboots" in l)
    assert "item=heavydutyboots" in rev


def test_no_battle_means_everything_assumed():
    sheet = render_sheet(default_state(), battle=None)
    assert "revealed: nothing yet" in sheet
    assert sheet.count("assumed(world-0 sample)") == 2


def test_our_side_never_marked_assumed():
    sheet = render_sheet(default_state(), battle())
    us = [l for l in sheet.splitlines() if l.startswith(("US", "  item"))]
    assert us and all("assumed" not in l for l in us)
    assert "US (sets exact)" in sheet


# ---- state axes --------------------------------------------------------

def test_boosts_status_counters_and_pp():
    ours = side([mon("kingambit", moves=[mv("suckerpunch", pp=6),
                                         mv("ironhead", pp=3,
                                            disabled=True)])],
                attack_boost=2, speed_boost=-1)
    sc = pe.SideConditions(toxic_count=3)
    theirs = pe.Side(pokemon=[mon("gliscor", status="toxic")],
                     side_conditions=sc)
    sheet = render_sheet(state(ours, theirs), battle())
    assert "boosts: atk+2 spe-1" in sheet
    assert "toxic(ctr=3)" in sheet
    assert "suckerpunch pp6" in sheet
    assert "ironhead pp3(disabled)" in sheet


def test_hazards_screens_field_and_trickroom():
    sc1 = pe.SideConditions(stealth_rock=1, spikes=2, reflect=3)
    ours = pe.Side(pokemon=[mon("kingambit")], side_conditions=sc1)
    theirs = side([mon("greattusk")])
    st = state(ours, theirs, weather=pe.Weather.SAND,
               weather_turns_remaining=3, trick_room=True,
               trick_room_turns_remaining=2)
    sheet = render_sheet(st, battle())
    assert "US hazards: SR=yes spikes=2 tspikes=0 web=no" in sheet
    assert "reflect=3t" in sheet
    assert "weather=sand(3t est)" in sheet
    assert "trickroom=yes(2t)" in sheet


def test_volatiles_substitute_and_last_move():
    ours = pe.Side(pokemon=[mon("kingambit")],
                   volatile_statuses={"substitute"}, substitute_health=61,
                   volatile_status_durations=pe.VolatileStatusDurations(
                       encore=2),
                   last_used_move="move:swordsdance")
    sheet = render_sheet(state(ours, side([mon("greattusk")])), battle())
    assert "substitute" in sheet and "subhp=61" in sheet
    assert "encore(2t)" in sheet
    assert "last_move=swordsdance" in sheet


def test_tera_availability_and_terastallized_flag():
    ours = side([mon("kingambit", terastallized=True, tera="dark")])
    theirs = side([mon("greattusk", tera="steel")])
    sheet = render_sheet(state(ours, theirs), battle())
    assert "TERA us=USED | them=available" in sheet
    assert "TERASTALLIZED=dark" in sheet
    # an unfired tera on their side stays in the assumed block
    assert "tera=steel" in next(l for l in sheet.splitlines()
                                if "assumed(world-0" in l)


def test_fainted_listed_from_battle_and_alive_counts():
    ours = side([mon("kingambit"), mon("dragonite", hp=0)])
    b = battle(team=[bmon("kingambit"), bmon("dragonite", fainted=True)],
               opp=[bmon("greattusk")])
    sheet = render_sheet(state(ours, side([mon("greattusk")])), b)
    # denominator is the format's fixed 6; the NAME comes from battle,
    # because the engine's fainted slot is a species-less dummy
    assert "ALIVE us=1/6 (fainted: dragonite)" in sheet
    assert not any("dragonite" in l for l in sheet.splitlines()
                   if l.startswith("US bench"))


# ---- beliefs -----------------------------------------------------------

class FakeObs:
    def forbidden(self, sp):
        return frozenset({"leftovers", "blacksludge"}) \
            if sp == "greattusk" else frozenset()

    def boots_inferred(self, sp):
        return "boots inferred (immune to SR on entry)" \
            if sp == "gholdengo" else None


def test_beliefs_section_only_when_obs_given():
    st = default_state()
    assert "BELIEFS" not in render_sheet(st, battle())
    sheet = render_sheet(st, battle(), obs=FakeObs())
    assert "greattusk: not blacksludge/leftovers" in sheet
    assert "gholdengo: boots inferred" in sheet


def test_beliefs_explicit_when_empty():
    class Empty:
        def forbidden(self, sp):
            return frozenset()

        def boots_inferred(self, sp):
            return None

    sheet = render_sheet(default_state(), battle(), obs=Empty())
    assert "none recorded" in sheet


# ---- engine round-trip artifacts ---------------------------------------
# State.from_string yields UPPERCASE ids/status, 'NONE' filler move slots,
# and last_used_move as a raw switch slot index — all found by rendering a
# real logged w0_state, all invisible to constructor-built states.

def test_uppercase_status_and_none_moves_normalized():
    from showdown import state_sheet as ss
    m = SimpleNamespace(status="NONE", rest_turns=0, sleep_turns=0)
    s = SimpleNamespace(side_conditions=SimpleNamespace(toxic_count=0))
    assert ss._status(m, s) == "none"
    moves = [SimpleNamespace(id="SUCKERPUNCH", pp=6, disabled=False),
             SimpleNamespace(id="NONE", pp=16, disabled=False)]
    assert ss._moves_with_pp(SimpleNamespace(moves=moves)) == \
        ["suckerpunch pp6"]


def test_last_move_switch_index_resolved_to_species():
    from showdown import state_sheet as ss
    mons = [SimpleNamespace(id="URSALUNA"), SimpleNamespace(id="RILLABOOM")]
    assert ss._last_move(SimpleNamespace(last_used_move="switch:1"),
                         mons) == "switch(rillaboom)"
    assert ss._last_move(SimpleNamespace(last_used_move="move:none"),
                         mons) is None
    assert ss._last_move(SimpleNamespace(last_used_move="move:ICEBEAM"),
                         mons) == "icebeam"


def test_placeholder_spread_suppressed_real_spread_shown():
    # the translator bakes spreads into raw stats and leaves evs/nature at
    # constructor defaults; rendering those as an assumed spread would be
    # a laundered placeholder. greattusk has a real spread, gholdengo the
    # defaults.
    sheet = render_sheet(default_state(), battle=None)
    lines = sheet.splitlines()
    tusk = lines[next(i for i, l in enumerate(lines)
                      if "THEM active: greattusk" in l) + 2]
    ghold = lines[next(i for i, l in enumerate(lines)
                       if "THEM bench: gholdengo" in l) + 2]
    assert "spread=jolly 0/252/0/0/4/252" in tusk
    assert "spread=" not in ghold
    assert "speed=" in ghold


# ---- overlay integration ----------------------------------------------

def test_turn_message_prefers_sheet_and_drops_appendix():
    from showdown.overlay import OverlayShadow
    rec = {"turn": 5, "reasons": ["opening"], "engine_choice": "suckerpunch",
           "engine_margin": 0.2, "worlds": [], "appendix": {"turn": 5},
           "sheet": "=== BOARD turn=5 ==="}
    msg = OverlayShadow._turn_message(rec)
    assert msg.startswith("=== BOARD turn=5 ===")
    assert "appendix" not in msg
    # fallback: no sheet -> the appendix JSON rides along as before
    rec2 = dict(rec)
    del rec2["sheet"]
    assert '"appendix"' in OverlayShadow._turn_message(rec2)
