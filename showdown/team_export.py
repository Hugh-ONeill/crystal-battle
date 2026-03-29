# export our engine teams to Showdown paste format
#
# Gen 2 format: species + level + IVs (for HP) + moves
# Items are supported but our engine doesn't track them
# Showdown uses IVs 0-31; Gen 2 DVs 0-15 map as IV = DV * 2

from __future__ import annotations

from engine.pokemon import Pokemon, PERFECT_DV


def _move_name_showdown(slot) -> str:
    """Get Showdown-format move name, with HP type annotation."""
    name = slot.template.name
    if slot.template.id == 237:  # Hidden Power
        hp_type = slot.template.type
        if hp_type and hp_type != "normal":
            return f"Hidden Power [{hp_type.title()}]"
    return name


def _dvs_to_ivs_line(dvs: dict[str, int]) -> str | None:
    """Convert our Gen 2 DVs to Showdown IV line, only if non-default."""
    # default is all 15 (perfect) -> all 30 in Showdown IVs
    if all(v == PERFECT_DV for v in dvs.values()):
        return None

    atk = dvs.get("attack", PERFECT_DV)
    dfn = dvs.get("defense", PERFECT_DV)
    spe = dvs.get("speed", PERFECT_DV)
    spc = dvs.get("special", PERFECT_DV)

    # Gen 2: HP DV is derived from the low bit of each other DV
    hp_dv = ((atk & 1) << 3) | ((dfn & 1) << 2) | ((spe & 1) << 1) | (spc & 1)

    parts = []
    # Showdown stat order: HP, Atk, Def, SpA, SpD, Spe
    # Gen 2 special DV controls both SpA and SpD
    stat_dvs = [
        ("HP", hp_dv),
        ("Atk", atk),
        ("Def", dfn),
        ("SpA", spc),    # showdown SpA maps to gen2 special
        ("SpD", spc),    # same DV for both
        ("Spe", spe),
    ]
    for sd_name, dv_val in stat_dvs:
        iv_val = dv_val * 2
        if iv_val != 30:  # only include non-default
            parts.append(f"{iv_val} {sd_name}")

    if not parts:
        return None
    return "IVs: " + " / ".join(parts)


def pokemon_to_showdown(mon: Pokemon) -> str:
    """Convert one of our Pokemon to Showdown paste format."""
    lines = [mon.species.name]
    lines.append("Level: 100")

    # include IVs if non-default (for Hidden Power typing)
    iv_line = _dvs_to_ivs_line(mon.dvs)
    if iv_line:
        lines.append(iv_line)

    for slot in mon.move_slots:
        lines.append(f"- {_move_name_showdown(slot)}")
    return "\n".join(lines)


def team_to_showdown(team: list[Pokemon]) -> str:
    """Convert a full team to Showdown paste format."""
    blocks = [pokemon_to_showdown(mon) for mon in team]
    return "\n\n".join(blocks)
