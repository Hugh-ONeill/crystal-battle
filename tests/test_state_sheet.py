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


# =====================================================================
# CASTER RENDERING
# =====================================================================
# The contract here is the inverse of the sheet above: the shadow LLM is
# handed world-0's guesses so it can judge them, and the casters must never
# see one, because a character asserts what a reasoner would weigh.

import poke_engine as pe

from showdown.state_sheet import render_caster_sheet


def _caster_fixture():
    """Their Zapdos has shown Hurricane and Roost; world-0 has SAMPLED its
    item, ability, tera and two more moves. None of the sample may appear."""
    ours = [mon("ironcrown", hp=134, maxhp=281, item="boosterenergy",
                ability="quarkdrive", status="brn",
                moves=["tachyoncutter", "voltswitch"]),
            mon("gliscor", item="toxicorb", ability="poisonheal",
                moves=["earthquake"])]
    theirs = [mon("zapdos", hp=81, maxhp=100, item="heavydutyboots",
                  ability="static", tera="steel",
                  moves=["hurricane", "roost", "voltswitch", "heatwave"]),
              mon("slowkinggalar", moves=["futuresight"])]
    s = state(side(ours), side(theirs))
    b = battle(turn=17, opp=[bmon("zapdos", moves=["hurricane", "roost"])])
    return s, b


def test_caster_sheet_never_leaks_a_sampled_set():
    """The measured failure this rendering exists to prevent: PRISM
    announced a Choice Scarf that no protocol line ever revealed, reasoning
    from a speed world-0 had sampled."""
    sheet = render_caster_sheet(*_caster_fixture())
    for sampled in ("Heavy-Duty Boots", "heavydutyboots", "Static", "static",
                    "Heat Wave", "heatwave"):
        assert sampled not in sheet, f"world-0's guess leaked: {sampled}"
    # their sampled TERA was Steel. A bare substring check cannot test that
    # any more, because real typing is on the board now and one of OUR mons
    # is Steel/Psychic — which is public fact, not a guess. What must hold
    # is that the tera stays unstated.
    assert "its Tera type" in sheet
    assert "Tera Steel" not in sheet and "pure Steel" not in sheet
    assert "Hurricane" in sheet and "Roost" in sheet, "reveals must survive"
    assert "NOT YET KNOWN" in sheet
    for gap in ("its item", "its ability", "its Tera type"):
        assert gap in sheet, f"the gap must be NAMED, not merely omitted: {gap}"


def test_caster_sheet_promotes_a_real_reveal_to_plain_fact():
    s, _ = _caster_fixture()
    b = battle(turn=17, opp=[bmon("zapdos", moves=["hurricane"],
                                  item="choicespecs", ability="static")])
    sheet = render_caster_sheet(s, battle=b)
    assert "Confirmed: item Choice Specs, ability Static." in sheet
    assert "its item" not in sheet and "its ability" not in sheet
    assert "its Tera type" in sheet, "still unrevealed, still named"


def test_caster_sheet_speaks_display_names_not_engine_ids():
    """Measured: engine-speak makes the characters free-associate. Every
    entity in the caster rendering is the name a human would say."""
    sheet = render_caster_sheet(*_caster_fixture())
    assert "Iron Crown" in sheet and "ironcrown" not in sheet
    assert "Tachyon Cutter" in sheet and "tachyoncutter" not in sheet
    assert "Booster Energy" in sheet and "boosterenergy" not in sheet
    assert "Quark Drive" in sheet and "quarkdrive" not in sheet
    assert "Slowking-Galar" in sheet and "slowkinggalar" not in sheet
    assert "burned" in sheet and "brn" not in sheet


def test_caster_sheet_field_words_match_the_beat_footer():
    """The sheet and the beat arrive in ONE prompt and the weather guard
    checks claims against the BEAT, so 'sun' here against 'harsh sun' there
    is a contradiction the model has to resolve on air."""
    ours = [mon("kingambit")]
    theirs = [mon("zapdos")]
    for token, expected in ((pe.Weather.SUN, "harsh sun"),
                            (pe.Weather.SAND, "a sandstorm"),
                            (pe.Weather.SNOW, "snow"),
                            (pe.Weather.HEAVY_RAIN, "heavy rain")):
        s = state(side(ours), side(theirs), weather=token)
        assert f"Weather: {expected}" in render_caster_sheet(s)
    s = state(side(ours), side(theirs), terrain=pe.Terrain.GRASSY)
    assert "Terrain: Grassy Terrain" in render_caster_sheet(s)
    # no turn counts: they hinge on unrevealed extender items and are the
    # engine's estimate, which is the laundering the collapse exists to stop
    s = state(side(ours), side(theirs), weather=pe.Weather.SNOW,
              weather_turns_remaining=5)
    assert "5" not in render_caster_sheet(s).split("US:")[0]


def test_caster_sheet_states_boosts_hazards_and_screens_in_words():
    ours = [mon("kingambit")]
    theirs = [mon("zapdos")]
    s = state(side(ours, side_conditions=pe.SideConditions(stealth_rock=1,
                                                           spikes=2)),
              side(theirs, side_conditions=pe.SideConditions(reflect=3),
                   attack_boost=2, speed_boost=-1))
    sheet = render_caster_sheet(s)
    assert "hazards: Stealth Rock, Spikes x2" in sheet
    assert "Reflect (3 turns left)" in sheet
    assert "Attack +2, Speed -1" in sheet


def test_caster_sheet_reports_tera_availability_both_ways():
    ours = [mon("kingambit", terastallized=True, tera="dark")]
    theirs = [mon("zapdos")]
    sheet = render_caster_sheet(state(side(ours), side(theirs)))
    assert "US: 1 alive, Tera ALREADY USED" in sheet
    assert "THEM: 1 alive, Tera still available" in sheet
    assert "TERASTALLIZED into Dark" in sheet


def test_caster_sheet_never_raises_and_is_deterministic():
    assert render_caster_sheet(object()) == ""
    assert render_caster_sheet(None, battle=None) == ""
    s, b = _caster_fixture()
    assert render_caster_sheet(s, battle=b) == render_caster_sheet(s, battle=b)


def test_caster_sheet_renders_no_events_only_state():
    """The beat text owns what HAPPENED. A sheet that also rendered the last
    move would give the model two accounts of one turn."""
    ours = [mon("kingambit")]
    theirs = [mon("zapdos")]
    s = state(side(ours, last_used_move="move:suckerpunch"), side(theirs))
    sheet = render_caster_sheet(s)
    assert "Sucker Punch" not in sheet and "last_move" not in sheet


# ---- the third register: reads the search worked out ---------------------

def _obs(confirmed=None, boots=(), ambiguous=(), forbidden=()):
    return SimpleNamespace(confirmed=dict(confirmed or {}),
                           boots=set(boots), boots_ambiguous=set(ambiguous),
                           forbidden=lambda sp: frozenset(forbidden))


def _reads_fixture():
    ours = [mon("ironcrown")]
    theirs = [mon("zapdos", hp=81, maxhp=100, item="choicescarf",
                  moves=["hurricane", "roost", "voltswitch"]),
              mon("slowkinggalar")]
    s = state(side(ours), side(theirs))
    b = battle(turn=21, opp=[bmon("zapdos", moves=["hurricane", "roost"])])
    return s, b


def test_a_confirmed_inference_is_its_own_register_not_an_unknown():
    """_emit_belief_deltas calls the Scarf ONCE, with its evidence chain. A
    board that then reports the item unknown on every later turn contradicts
    a call the desk already made on air, inside the same prompt."""
    s, b = _reads_fixture()
    sheet = render_caster_sheet(s, battle=b,
                                obs=_obs({"zapdos": "choicescarf"}))
    assert "READ FROM PLAY" in sheet and "Choice Scarf" in sheet
    assert "its item" not in sheet, "a called Scarf is not an unknown item"
    assert "Never say it was revealed" in sheet
    # the other axes are untouched by an item read
    assert "its ability, its Tera type" in sheet


def test_a_read_that_later_evidence_ruled_out_is_never_spoken():
    """Mirrors the filter _log_inferred_items applies before persisting: an
    inference the constraint layer killed must not be said either."""
    s, b = _reads_fixture()
    sheet = render_caster_sheet(
        s, battle=b, obs=_obs({"zapdos": "choicescarf"},
                              forbidden=("choicescarf",)))
    assert "READ FROM PLAY" not in sheet and "Choice Scarf" not in sheet
    assert "NOT YET KNOWN — its item" in sheet


def test_a_real_reveal_outranks_a_read():
    s, _ = _reads_fixture()
    b = battle(turn=21, opp=[bmon("zapdos", moves=["hurricane"],
                                  item="leftovers")])
    sheet = render_caster_sheet(s, battle=b,
                                obs=_obs({"zapdos": "choicescarf"}))
    assert "Confirmed: item Leftovers" in sheet
    assert "READ FROM PLAY" not in sheet, "protocol beats inference"


def test_boots_reads_are_confident_or_hedged_by_magic_guard():
    s, b = _reads_fixture()
    confident = render_caster_sheet(s, battle=b, obs=_obs(boots=("zapdos",)))
    assert "READ FROM PLAY" in confident and "Heavy-Duty Boots" in confident
    hedged = render_caster_sheet(s, battle=b, obs=_obs(ambiguous=("zapdos",)))
    assert "could be Magic Guard instead" in hedged, \
        "set_inference records the ambiguous case exactly so a caster hedges"


def test_a_read_follows_a_mon_to_the_bench():
    """Where the sheet earns most: the set_reveal beat fired once, maybe
    fifteen turns ago, and is long gone from the transcript and the beat
    window. Restating it is the only way a bench callout is ever sayable."""
    s, b = _reads_fixture()
    sheet = render_caster_sheet(
        s, battle=b, obs=_obs({"slowkinggalar": "assaultvest"},
                              boots=("slowkinggalar",)))
    bench = next(l for l in sheet.splitlines() if "Their bench" in l)
    assert "Slowking-Galar (Poison/Psychic) 100% [read: Assault Vest]" in bench


def test_reads_are_absent_without_obs_and_never_come_from_world_zero():
    """No obs -> no reads, and world-0's sampled Choice Scarf on their Zapdos
    still never appears. The sample is not evidence in any register."""
    s, b = _reads_fixture()
    sheet = render_caster_sheet(s, battle=b)
    assert "READ FROM PLAY" not in sheet
    assert "Choice Scarf" not in sheet and "choicescarf" not in sheet
    assert "NOT YET KNOWN — its item" in sheet


def test_pending_wish_rendered_when_live():
    ours = side([mon("kingambit")])
    theirs = side([mon("greattusk")], wish=(1, 212))
    sheet = render_sheet(state(ours, theirs), battle())
    assert "WISH them: 212 hp arrives at end of THIS turn" in sheet
    assert "WISH us" not in sheet
    quiet = render_sheet(default_state(), battle())
    assert "WISH" not in quiet


def test_pending_future_sight_rendered_with_caster():
    ours = side([mon("kingambit")])
    theirs = side([mon("slowkinggalar")], future_sight=(2, "0"))
    sheet = render_sheet(state(ours, theirs), battle())
    assert "FUTURE SIGHT by them (slowkinggalar): lands end of NEXT turn" \
        in sheet
    assert "FUTURE SIGHT by us" not in sheet


def test_the_board_states_typing():
    """A hand-flagged line said "Zapdos will have to account for those
    Spikes every time it returns". Zapdos is Flying and immune, and nothing
    the caster was given said so — the board carried hp, status, item,
    ability and moves, and no typing. The RAG cannot cover this either: it
    routes on curated mechanic names and a type matchup is not one."""
    ours = [mon("gliscor")]
    theirs = [mon("zapdos")]
    sheet = render_caster_sheet(state(side(ours), side(theirs)))
    assert "Zapdos (Electric/Flying)" in sheet
    assert "Gliscor (Ground/Flying)" in sheet


def test_a_terastallized_mon_shows_its_tera_type_not_its_dex_typing():
    """Tera REPLACES typing, so printing the dex entry would state a typing
    that has left the field — the same reason the type-claim guard keys on
    _tera rather than the pokedex."""
    theirs = [mon("dondozo", terastallized=True, tera="dark")]
    sheet = render_caster_sheet(state(side([mon("kingambit")]), side(theirs)))
    assert "now pure Dark" in sheet
    assert "Dondozo (Water)" not in sheet


def test_bench_mons_carry_typing_too():
    """Hazard immunity is a question about who can come IN."""
    theirs = [mon("greattusk"), mon("zapdos"), mon("gholdengo")]
    sheet = render_caster_sheet(state(side([mon("kingambit")]), side(theirs)))
    bench = next(l for l in sheet.splitlines() if "Their bench" in l)
    assert "Zapdos (Electric/Flying)" in bench
