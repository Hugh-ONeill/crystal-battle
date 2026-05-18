#!/usr/bin/env python3
"""Smoke tests for poke-engine mechanic coverage.

Each test builds a state with one mechanic-relevant team setup, generates
instructions for a chosen action pair, and inspects the returned instruction
list for the mechanic-specific effect that *should* fire (FormeChange,
ChangeAbility, ChangeType, ApplyVolatileStatus encore, SetFutureSight,
ChangeItem, ToggleTrickRoom, damage = exactly 1/8 maxhp, Earthquake hits
Levitate target, etc.).

Sections:
  - FORM-CHANGE ABILITIES: Imposter, Illusion, Stance Change, Tera Shift,
    Disguise, Ice Face.
  - MOVE-EFFECT HANDLERS: Magic Coat, Encore, Sucker Punch (twin: vs status
    and vs attack), Future Sight, Knock Off.
  - FIELD / ABILITY-SUPPRESSION: Trick Room, Gravity, Mold Breaker.

Status legend in output:
  PASS     — engine produced the expected mechanic-specific effect
  SILENT   — engine emitted no relevant instructions (mechanic absent)
  PARTIAL  — engine emitted something, but not the right shape
  ERROR    — couldn't build state / generate instructions

Usage:
  .venv/bin/python tools/engine_smoke_tests.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9


# ============================================================
# HELPERS
# ============================================================

@dataclass
class Result:
    name: str
    status: str  # PASS / SILENT / PARTIAL / ERROR
    detail: str


def _instr_reprs(branches) -> list[str]:
    """Flatten all instruction reprs across branches into a single list."""
    out: list[str] = []
    for br in branches:
        for op in br.instruction_list:
            out.append(repr(op))
    return out


def _contains_any(reprs: list[str], needles: list[str]) -> bool:
    return any(any(n in r for n in needles) for r in reprs)


def _safely(fn, name: str) -> Result:
    try:
        return fn()
    except Exception as e:
        return Result(name=name, status="ERROR", detail=f"{type(e).__name__}: {e}")


# ============================================================
# FORM-CHANGE ABILITIES
# ============================================================

def test_imposter() -> Result:
    """Ditto switches in on the opp's Garchomp; expected outcome is a
    transform that copies species/stats/ability/moves. We check for
    ChangeAbility / ChangeType / ChangeAttack — at minimum the ability
    should change from 'imposter' to 'roughskin' on switch-in."""
    team1 = """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
EVs: 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch

Ditto @ Choice Scarf
Ability: Imposter
Tera Type: Normal
EVs: 252 HP / 252 SpA / 4 Spe
Modest Nature
IVs: 0 Atk
- Transform"""
    team2 = """Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Outrage
- Stone Edge
- Swords Dance"""
    state = build_pe_state_gen9(team1, team2)
    # Switch Ditto in; Garchomp does Swords Dance (no damage to noise).
    instr = pe.generate_instructions(state, "ditto", "swordsdance")
    reprs = _instr_reprs(instr)
    transform_signals = [
        "ChangeAbility", "ChangeType", "ChangeAttack", "ChangeSpeed",
        "ChangeSpecialAttack", "ChangeSpecialDefense", "ChangeDefense",
        "FormeChange",
    ]
    if _contains_any(reprs, transform_signals):
        return Result("Imposter (Ditto)", "PASS",
                      f"transform-related instruction found in {len(reprs)} ops")
    only_switch_boost = all(
        ("Switch" in r) or ("Boost" in r) or ("SetLastUsedMove" in r)
        or ("ChangeSideCondition" in r) or ("DecrementPP" in r)
        for r in reprs
    )
    status = "SILENT" if only_switch_boost else "PARTIAL"
    return Result("Imposter (Ditto)", status,
                  f"no transform signals in {len(reprs)} ops: "
                  f"{[r.split(':')[0] for r in reprs]}")


def test_illusion() -> Result:
    """Zoroark-Hisui as the lead, Garchomp in the last slot. With proper
    Illusion, opp's view of the active mon should show as Garchomp until
    Zoroark takes a damaging hit. The engine has no perfect-vs-imperfect
    info distinction, but we can at least check whether ANY illusion-
    related instruction (FormeChange, ChangeType disguise→true) fires when
    Zoroark switches in or is hit."""
    team1 = """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
- Volt Switch

Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
- Earthquake

Zoroark-Hisui @ Choice Specs
Ability: Illusion
Tera Type: Ghost
EVs: 252 SpA / 252 Spe
Timid Nature
- Shadow Ball"""
    team2 = """Tyranitar @ Smooth Rock
Ability: Sand Stream
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Adamant Nature
- Crunch"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "zoroarkhisui", "crunch")
    reprs = _instr_reprs(instr)
    # Look for any indication of a disguise-state change.
    disguise_signals = ["FormeChange", "ChangeType", "Illusion"]
    if _contains_any(reprs, disguise_signals):
        # Could be a real Illusion break or a coincidental ChangeType from a
        # move — inspect the detail to disambiguate.
        return Result("Illusion (Zoroark-H)", "PARTIAL",
                      f"found disguise-shaped instruction; needs hand-review: {reprs}")
    return Result("Illusion (Zoroark-H)", "SILENT",
                  f"no disguise signals in {len(reprs)} ops; Zoroark observable as itself")


def test_stance_change() -> Result:
    """Aegislash in Shield form uses Shadow Sneak. Expected: FormeChange
    to Blade form before the move resolves."""
    team1 = """Aegislash @ Leftovers
Ability: Stance Change
Tera Type: Ghost
EVs: 252 HP / 252 Atk
Adamant Nature
- Shadow Sneak
- King's Shield
- Iron Head
- Swords Dance"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "shadowsneak", "softboiled")
    reprs = _instr_reprs(instr)
    if _contains_any(reprs, ["FormeChange"]):
        return Result("Stance Change (Aegislash)", "PASS",
                      "FormeChange instruction fired")
    return Result("Stance Change (Aegislash)", "SILENT",
                  f"no FormeChange in {len(reprs)} ops")


def test_tera_shift() -> Result:
    """Switch a fresh Terapagos in. Expected: an immediate FormeChange
    or ChangeType to Terapagos-Terastal as soon as it hits the field."""
    team1 = """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
- Volt Switch

Terapagos @ Leftovers
Ability: Tera Shift
Tera Type: Stellar
EVs: 252 HP / 252 SpA
Modest Nature
- Tera Starstorm
- Earth Power"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "terapagos", "softboiled")
    reprs = _instr_reprs(instr)
    if _contains_any(reprs, ["FormeChange", "ChangeType"]):
        return Result("Tera Shift (Terapagos)", "PASS",
                      "form-change fired on switch-in")
    return Result("Tera Shift (Terapagos)", "SILENT",
                  f"no form change in {len(reprs)} ops; Terapagos stays base")


def test_disguise() -> Result:
    """Mimikyu takes Earthquake from Garchomp. Expected: damage equals
    exactly 1/8 maxhp instead of the full ~150+ calc, and a FormeChange
    instruction transitions Mimikyu → Mimikyu-Busted."""
    team1 = """Mimikyu @ Life Orb
Ability: Disguise
Tera Type: Ghost
EVs: 252 Atk / 252 Spe
Jolly Nature
- Play Rough
- Swords Dance"""
    team2 = """Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Earthquake"""
    state = build_pe_state_gen9(team1, team2)
    mimikyu_maxhp = state.side_one.pokemon[0].maxhp
    expected_disguise_dmg = mimikyu_maxhp // 8
    instr = pe.generate_instructions(state, "playrough", "earthquake")
    reprs = _instr_reprs(instr)
    # Find the damage dealt to SideOne (Mimikyu).
    side_one_dmg = None
    for r in reprs:
        if r.startswith("Damage SideOne"):
            try:
                side_one_dmg = int(r.split(":")[-1].strip())
            except ValueError:
                pass
            break
    has_forme_change = _contains_any(reprs, ["FormeChange"])
    if (side_one_dmg is not None
            and abs(side_one_dmg - expected_disguise_dmg) <= 2
            and has_forme_change):
        return Result("Disguise (Mimikyu)", "PASS",
                      f"dmg={side_one_dmg} (~maxhp/8={expected_disguise_dmg}) + FormeChange")
    if has_forme_change and side_one_dmg is None:
        return Result("Disguise (Mimikyu)", "PARTIAL",
                      "FormeChange fired but no damage line found")
    if has_forme_change:
        return Result("Disguise (Mimikyu)", "PARTIAL",
                      f"FormeChange fired but dmg={side_one_dmg} ≠ maxhp/8={expected_disguise_dmg}")
    if side_one_dmg is not None and side_one_dmg > expected_disguise_dmg * 2:
        return Result("Disguise (Mimikyu)", "SILENT",
                      f"full damage taken (dmg={side_one_dmg}, "
                      f"expected disguise={expected_disguise_dmg}); no FormeChange")
    return Result("Disguise (Mimikyu)", "SILENT",
                  f"dmg={side_one_dmg}, no FormeChange; full ops: {reprs}")


def test_ice_face() -> Result:
    """Eiscue takes Close Combat (physical). Expected: damage 0, FormeChange
    to Eiscue-Noice."""
    team1 = """Eiscue @ Leftovers
Ability: Ice Face
Tera Type: Ice
EVs: 252 HP / 252 Def
Impish Nature
- Liquidation
- Belly Drum"""
    team2 = """Iron Hands @ Booster Energy
Ability: Quark Drive
Tera Type: Fighting
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Close Combat"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "liquidation", "closecombat")
    reprs = _instr_reprs(instr)
    # When the disguise absorbs the hit, no Damage SideOne instruction fires.
    side_one_took_damage = any(r.startswith("Damage SideOne") for r in reprs)
    has_forme_change = _contains_any(reprs, ["FormeChange"])
    if has_forme_change and not side_one_took_damage:
        return Result("Ice Face (Eiscue)", "PASS",
                      "FormeChange + no damage on Eiscue (disguise absorbed physical hit)")
    if has_forme_change:
        return Result("Ice Face (Eiscue)", "PARTIAL",
                      "FormeChange fired but Eiscue still took damage")
    return Result("Ice Face (Eiscue)", "SILENT",
                  f"no FormeChange; Eiscue {'took damage' if side_one_took_damage else 'no damage'}")


# ============================================================
# MOVE-EFFECT HANDLERS
# ============================================================

def test_magic_coat() -> Result:
    """User uses Magic Coat, opp uses Toxic. Expected: the Toxic status
    ends up on the opp (reflected) instead of the user."""
    team1 = """Smeargle @ Focus Sash
Ability: Own Tempo
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 Spe
Timid Nature
- Magic Coat
- Spore"""
    team2 = """Toxapex @ Black Sludge
Ability: Regenerator
Tera Type: Steel
EVs: 252 HP / 252 Def
Bold Nature
- Toxic
- Recover"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "magiccoat", "toxic")
    reprs = _instr_reprs(instr)
    # If reflected correctly: ChangeStatus targets SideTwo (opp got toxic'd)
    # If silent: ChangeStatus targets SideOne (Smeargle got toxic'd)
    on_user = any("ChangeStatus SideOne" in r for r in reprs)
    on_opp = any("ChangeStatus SideTwo" in r for r in reprs)
    if on_opp and not on_user:
        return Result("Magic Coat", "PASS",
                      "Toxic reflected onto opp (ChangeStatus SideTwo)")
    if on_user and not on_opp:
        return Result("Magic Coat", "SILENT",
                      "Toxic landed on user (reflection failed)")
    return Result("Magic Coat", "PARTIAL",
                  f"unexpected status pattern: {[r for r in reprs if 'Status' in r]}")


def test_encore() -> Result:
    """Slow user (Clefable, no Prankster) uses Encore after opp uses Swords
    Dance. Expected: ApplyVolatileStatus encore on SideTwo."""
    team1 = """Clefable @ Leftovers
Ability: Magic Guard
Tera Type: Fairy
EVs: 252 HP / 252 Def
Bold Nature
- Encore
- Moonblast
- Soft-Boiled
- Calm Mind"""
    team2 = """Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Swords Dance"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "encore", "swordsdance")
    reprs = _instr_reprs(instr)
    if _contains_any(reprs, ["ApplyVolatileStatus SideTwo: Encore",
                              "ApplyVolatileStatus SideTwo: ENCORE",
                              "VolatileStatus SideTwo: Encore"]):
        return Result("Encore", "PASS",
                      "Encore volatile applied to opp")
    # be lenient — match any 'encore' substring on SideTwo
    has_encore_apply = any("Encore" in r and "SideTwo" in r and
                           "Apply" in r for r in reprs)
    if has_encore_apply:
        return Result("Encore", "PASS", "Encore-shaped volatile applied to opp")
    return Result("Encore", "SILENT",
                  f"no Encore volatile on opp; ops: {[r.split(':')[0] for r in reprs]}")


def test_sucker_punch_vs_status() -> Result:
    """Sucker Punch should FAIL when opp picks a status move. Expected:
    no Damage SideTwo instruction (Sucker Punch missed/failed)."""
    team1 = """Bisharp @ Life Orb
Ability: Defiant
Tera Type: Dark
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Sucker Punch
- Iron Head"""
    team2 = """Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Swords Dance"""
    state = build_pe_state_gen9(team1, team2)
    # opp picks a status move → Sucker Punch should fail
    instr = pe.generate_instructions(state, "suckerpunch", "swordsdance")
    reprs = _instr_reprs(instr)
    damaged_opp = any("Damage SideTwo" in r for r in reprs)
    if not damaged_opp:
        return Result("Sucker Punch (vs status)", "PASS",
                      "no damage on opp — Sucker Punch correctly failed vs SD")
    return Result("Sucker Punch (vs status)", "SILENT",
                  "Sucker Punch dealt damage despite opp using status — "
                  "conditional-priority logic missing")


def test_sucker_punch_vs_attack() -> Result:
    """Sucker Punch should HIT when opp picks an attacking move. Expected:
    Damage SideTwo present, and Bisharp's hit lands before opp's
    Earthquake (Sucker Punch has +1 priority)."""
    team1 = """Bisharp @ Life Orb
Ability: Defiant
Tera Type: Dark
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Sucker Punch
- Iron Head"""
    team2 = """Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Swords Dance"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "suckerpunch", "earthquake")
    reprs = _instr_reprs(instr)
    damaged_opp = any("Damage SideTwo" in r for r in reprs)
    if damaged_opp:
        return Result("Sucker Punch (vs attack)", "PASS",
                      "Sucker Punch hit opp who was attacking")
    return Result("Sucker Punch (vs attack)", "SILENT",
                  "no damage on opp despite opp picking attack")


def test_future_sight() -> Result:
    """User uses Future Sight. Expected: SetFutureSight instruction
    (delayed attack scheduled for two turns later)."""
    team1 = """Slowking @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 248 HP / 252 SpA / 8 SpD
Modest Nature
- Future Sight
- Slack Off"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "futuresight", "softboiled")
    reprs = _instr_reprs(instr)
    if _contains_any(reprs, ["SetFutureSight"]):
        return Result("Future Sight", "PASS",
                      "SetFutureSight instruction emitted (delayed attack scheduled)")
    return Result("Future Sight", "SILENT",
                  f"no SetFutureSight; ops: {[r.split(':')[0] for r in reprs]}")


def test_knock_off() -> Result:
    """User uses Knock Off on a Heavy-Duty Boots holder. Expected:
    ChangeItem SideTwo (item knocked off) + Damage SideTwo (with the
    1.5x Knock Off damage boost since target had an item)."""
    team1 = """Meowscarada @ Choice Band
Ability: Protean
Tera Type: Grass
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Knock Off
- Flower Trick"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "knockoff", "softboiled")
    reprs = _instr_reprs(instr)
    has_change_item = _contains_any(reprs, ["ChangeItem SideTwo"])
    has_damage = any("Damage SideTwo" in r for r in reprs)
    if has_change_item and has_damage:
        return Result("Knock Off", "PASS",
                      "Damage + ChangeItem on opp (item removed)")
    if has_damage and not has_change_item:
        return Result("Knock Off", "PARTIAL",
                      "damage dealt but item not removed")
    return Result("Knock Off", "SILENT",
                  f"no Knock Off effect; ops: {[r.split(':')[0] for r in reprs]}")


# ============================================================
# FIELD / ABILITY-SUPPRESSION
# ============================================================

def test_trick_room() -> Result:
    """User uses Trick Room. Expected: ToggleTrickRoom instruction (or
    ChangeTrickRoom — either signals the field flip is wired up)."""
    team1 = """Hatterene @ Leftovers
Ability: Magic Bounce
Tera Type: Water
EVs: 252 HP / 252 SpA
Quiet Nature
IVs: 0 Spe
- Trick Room
- Psyshock
- Draining Kiss"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "trickroom", "softboiled")
    reprs = _instr_reprs(instr)
    if _contains_any(reprs, ["ToggleTrickRoom", "ChangeTrickRoom"]):
        return Result("Trick Room", "PASS",
                      "ToggleTrickRoom instruction emitted")
    return Result("Trick Room", "SILENT",
                  f"no TR toggle; ops: {[r.split(':')[0] for r in reprs]}")


def test_mold_breaker() -> Result:
    """Haxorus with Mold Breaker uses Earthquake on Rotom-Heat (Levitate).
    With Mold Breaker working, Levitate is bypassed and EQ deals damage;
    without, EQ does 0 damage (Levitate blocks)."""
    team1 = """Haxorus @ Life Orb
Ability: Mold Breaker
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Dragon Claw"""
    team2 = """Rotom-Heat @ Heavy-Duty Boots
Ability: Levitate
Tera Type: Water
EVs: 248 HP / 8 Def / 252 SpA
Modest Nature
IVs: 0 Atk
- Volt Switch
- Overheat"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "earthquake", "voltswitch")
    reprs = _instr_reprs(instr)
    # Any Damage SideTwo line (with non-trivial damage) means Levitate
    # was bypassed by Mold Breaker.
    eq_damage = None
    for r in reprs:
        if r.startswith("Damage SideTwo"):
            try:
                eq_damage = int(r.split(":")[-1].strip())
                break
            except ValueError:
                pass
    if eq_damage is not None and eq_damage > 0:
        return Result("Mold Breaker", "PASS",
                      f"EQ dealt {eq_damage} to Levitate target (ability bypassed)")
    if eq_damage == 0:
        return Result("Mold Breaker", "PARTIAL",
                      "EQ instruction present but damage=0 (Levitate not bypassed)")
    return Result("Mold Breaker", "SILENT",
                  "no EQ damage on Levitate target — Mold Breaker silent")


# ============================================================
# MULTI-TURN / MULTI-HIT / PIVOT / FIELD
# ============================================================

def test_outrage_lockin() -> Result:
    """Outrage should apply a LOCKEDMOVE volatile to the user and disable
    the user's other moves for the lock duration."""
    team1 = """Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Outrage
- Earthquake
- Stone Edge
- Swords Dance"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "outrage", "softboiled")
    reprs = _instr_reprs(instr)
    has_lockedmove = any("LOCKEDMOVE" in r and "ApplyVolatileStatus SideOne" in r
                         for r in reprs)
    has_disables = sum(1 for r in reprs if r.startswith("DisableMove SideOne"))
    if has_lockedmove and has_disables >= 3:
        return Result("Outrage lock-in", "PASS",
                      f"LOCKEDMOVE volatile applied + {has_disables} moves disabled")
    if has_lockedmove:
        return Result("Outrage lock-in", "PARTIAL",
                      f"LOCKEDMOVE applied but only {has_disables} moves disabled")
    return Result("Outrage lock-in", "SILENT",
                  f"no LOCKEDMOVE volatile; ops: {[r.split(':')[0] for r in reprs]}")


def test_solar_beam_charging() -> Result:
    """Solar Beam without sun should charge turn 1 (apply SOLARBEAM volatile,
    deal no damage), then fire turn 2. Verify turn-1 produces the charge
    volatile and no Damage SideTwo."""
    team1 = """Venusaur @ Life Orb
Ability: Chlorophyll
Tera Type: Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Solar Beam"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "solarbeam", "softboiled")
    reprs = _instr_reprs(instr)
    has_charge = any("SOLARBEAM" in r and "ApplyVolatileStatus SideOne" in r
                     for r in reprs)
    dealt_damage = any(r.startswith("Damage SideTwo") for r in reprs)
    if has_charge and not dealt_damage:
        return Result("Solar Beam charging", "PASS",
                      "SOLARBEAM charge volatile applied, no turn-1 damage")
    if dealt_damage and not has_charge:
        return Result("Solar Beam charging", "SILENT",
                      "fired immediately without charging (sun-only behaviour leaked)")
    return Result("Solar Beam charging", "PARTIAL",
                  f"unexpected pattern: charge={has_charge} damage={dealt_damage}")


def test_bullet_seed_multihit() -> Result:
    """Multi-hit moves should emit multiple Damage SideTwo instructions
    per branch (one per hit) and have probabilistic branches by hit count."""
    team1 = """Breloom @ Life Orb
Ability: Technician
Tera Type: Grass
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Bullet Seed"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    branches = pe.generate_instructions(state, "bulletseed", "softboiled")
    hit_counts = []
    for br in branches:
        hits = sum(1 for op in br.instruction_list if repr(op).startswith("Damage SideTwo"))
        hit_counts.append(hits)
    max_hits = max(hit_counts) if hit_counts else 0
    if len(branches) >= 2 and max_hits >= 2:
        return Result("Bullet Seed multi-hit", "PASS",
                      f"{len(branches)} branches with hit counts {hit_counts}")
    if max_hits == 1:
        return Result("Bullet Seed multi-hit", "SILENT",
                      "single-hit only — multi-hit logic absent")
    return Result("Bullet Seed multi-hit", "PARTIAL",
                  f"{len(branches)} branches, hit_counts={hit_counts}")


def test_sticky_web_set() -> Result:
    """Sticky Web should emit ChangeSideCondition StickyWeb on the
    opponent's side."""
    team1 = """Smeargle @ Focus Sash
Ability: Own Tempo
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 Spe
Timid Nature
- Sticky Web"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "stickyweb", "softboiled")
    reprs = _instr_reprs(instr)
    if any("StickyWeb" in r and "ChangeSideCondition SideTwo" in r for r in reprs):
        return Result("Sticky Web", "PASS",
                      "StickyWeb side condition set on opponent")
    return Result("Sticky Web", "SILENT",
                  f"no StickyWeb side condition; ops: {[r.split(':')[0] for r in reprs]}")


def test_uturn_pivot() -> Result:
    """U-turn should deal damage AND trigger a force-switch on the user's
    side (ToggleSideOneForceSwitch). Requires a switch target on the bench."""
    team1 = """Tornadus-Therian @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- U-turn

Garchomp @ Life Orb
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 252 Spe
Jolly Nature
- Earthquake"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "uturn", "softboiled")
    reprs = _instr_reprs(instr)
    has_damage = any(r.startswith("Damage SideTwo") for r in reprs)
    has_force_switch = any("ToggleSideOneForceSwitch" in r for r in reprs)
    if has_damage and has_force_switch:
        return Result("U-turn pivot", "PASS",
                      "damage dealt + ToggleSideOneForceSwitch emitted")
    if has_damage and not has_force_switch:
        return Result("U-turn pivot", "SILENT",
                      "damage only — pivot effect missing")
    return Result("U-turn pivot", "PARTIAL",
                  f"damage={has_damage} switch={has_force_switch}")


def test_wonder_room() -> Result:
    """Wonder Room should swap Def and SpD for all mons while active.
    Engine has no dedicated instruction for it — look for ANY state change
    in the instruction list beyond trivial bookkeeping."""
    team1 = """Slowking @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
IVs: 0 Atk
- Wonder Room"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    instr = pe.generate_instructions(state, "wonderroom", "softboiled")
    reprs = _instr_reprs(instr)
    # Anything that mentions wonder room, or any swap of def/spd stats
    has_wonder_signal = any(
        "WonderRoom" in r or "WONDERROOM" in r or "wonderroom" in r
        for r in reprs
    )
    if has_wonder_signal:
        return Result("Wonder Room", "PASS",
                      "Wonder Room state change emitted")
    # If the only effects are opponent's move + tail bookkeeping, the
    # engine treated Wonder Room as a no-op.
    only_passive = all(
        ("Heal" in r) or ("ChangeStatus" in r) or ("DecrementPP" in r)
        or ("SetLastUsedMove" in r) or ("Damage" in r and "SideTwo" not in r)
        for r in reprs
    )
    status = "SILENT" if only_passive else "PARTIAL"
    return Result("Wonder Room", status,
                  f"no Wonder-Room state change; ops: {[r.split(':')[0] for r in reprs]}")


def test_belly_drum() -> Result:
    """Belly Drum should deal 50% maxhp self-damage and set Attack to +6."""
    team1 = """Azumarill @ Sitrus Berry
Ability: Huge Power
Tera Type: Water
EVs: 252 HP / 252 Atk
Adamant Nature
- Belly Drum"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    maxhp = state.side_one.pokemon[0].maxhp
    expected_self_dmg = maxhp // 2
    instr = pe.generate_instructions(state, "bellydrum", "softboiled")
    reprs = _instr_reprs(instr)
    self_dmg = None
    for r in reprs:
        if r.startswith("Damage SideOne"):
            try:
                self_dmg = int(r.split(":")[-1].strip())
                break
            except ValueError:
                pass
    has_max_boost = any("Boost SideOne Attack: 6" in r for r in reprs)
    if has_max_boost and self_dmg is not None and abs(self_dmg - expected_self_dmg) <= 2:
        return Result("Belly Drum", "PASS",
                      f"dmg={self_dmg} (~maxhp/2={expected_self_dmg}) + Attack +6")
    if has_max_boost:
        return Result("Belly Drum", "PARTIAL",
                      f"+6 boost applied but self-damage={self_dmg} (expected {expected_self_dmg})")
    return Result("Belly Drum", "SILENT",
                  f"no +6 Attack boost; ops: {[r.split(':')[0] for r in reprs]}")


def test_substitute() -> Result:
    """Substitute should deal 25% maxhp self-damage, set substitute health,
    and apply the SUBSTITUTE volatile."""
    team1 = """Garchomp @ Leftovers
Ability: Rough Skin
Tera Type: Steel
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Substitute"""
    team2 = """Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Normal
EVs: 252 HP / 252 Def
Bold Nature
- Soft-Boiled"""
    state = build_pe_state_gen9(team1, team2)
    maxhp = state.side_one.pokemon[0].maxhp
    expected_sub_hp = maxhp // 4
    instr = pe.generate_instructions(state, "substitute", "softboiled")
    reprs = _instr_reprs(instr)
    has_volatile = any("SUBSTITUTE" in r and "ApplyVolatileStatus SideOne" in r
                       for r in reprs)
    sub_hp = None
    for r in reprs:
        if r.startswith("ChangeSubstituteHealth SideOne"):
            try:
                sub_hp = int(r.split(":")[-1].strip())
                break
            except ValueError:
                pass
    if has_volatile and sub_hp is not None and abs(sub_hp - expected_sub_hp) <= 2:
        return Result("Substitute", "PASS",
                      f"volatile + sub_hp={sub_hp} (~maxhp/4={expected_sub_hp})")
    if has_volatile:
        return Result("Substitute", "PARTIAL",
                      f"volatile applied but sub_hp={sub_hp} (expected {expected_sub_hp})")
    return Result("Substitute", "SILENT",
                  f"no SUBSTITUTE volatile; ops: {[r.split(':')[0] for r in reprs]}")


# ============================================================
# DRIVER
# ============================================================

TESTS = [
    # form-change abilities
    ("Imposter (Ditto)", test_imposter),
    ("Illusion (Zoroark-H)", test_illusion),
    ("Stance Change (Aegislash)", test_stance_change),
    ("Tera Shift (Terapagos)", test_tera_shift),
    ("Disguise (Mimikyu)", test_disguise),
    ("Ice Face (Eiscue)", test_ice_face),
    # move-effect handlers
    ("Magic Coat", test_magic_coat),
    ("Encore", test_encore),
    ("Sucker Punch (vs status)", test_sucker_punch_vs_status),
    ("Sucker Punch (vs attack)", test_sucker_punch_vs_attack),
    ("Future Sight", test_future_sight),
    ("Knock Off", test_knock_off),
    # field / ability-suppression
    ("Trick Room", test_trick_room),
    ("Mold Breaker", test_mold_breaker),
    # multi-turn / multi-hit / pivot / field
    ("Outrage lock-in", test_outrage_lockin),
    ("Solar Beam charging", test_solar_beam_charging),
    ("Bullet Seed multi-hit", test_bullet_seed_multihit),
    ("Sticky Web", test_sticky_web_set),
    ("U-turn pivot", test_uturn_pivot),
    ("Wonder Room", test_wonder_room),
    ("Belly Drum", test_belly_drum),
    ("Substitute", test_substitute),
]


def main():
    print("=== poke-engine mechanic-coverage smoke tests ===\n")
    results = [_safely(fn, name) for name, fn in TESTS]
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {r.name:<{width}}  {r.status:<8}  {r.detail}")
    print()
    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = " / ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
