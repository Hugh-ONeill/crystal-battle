"""Deterministic board fact sheet: engine State -> text an LLM can trust.

WHY THIS EXISTS. The caster guard stack (caster.py's ~20 _*_claim validators)
is a catalogue of state fields the model was never given: every measured
hallucination class — invented weather, phantom screens, wrong-sign boosts,
narrating a long-fainted mon as active — was the model filling a gap in a
partial view, and every fix that worked was a structured fact stated as a
constraint. This module is that lesson applied at the source: render the
ENTIRE board once, deterministically, instead of guarding claim-by-claim
after the fact.

Three rules, each load-bearing:

1. EPISTEMIC REGISTER. Our side is exact (Showdown sends our real sets in
   the request). Their side is a MIX: the protocol reveals hp, status,
   boosts, faints and used moves, while the world-sampler GUESSED the rest
   (remaining moves, item, ability, spread, tera). A guess printed in the
   same register as a fact is how a prior gets laundered into a claim, so
   every opposing mon splits into `revealed:` and `assumed:` lines and the
   assumed line never mixes into the revealed one.

2. FIXED SLOTS, EXPLICIT ABSENCE. Every axis prints every turn — "hazards:
   none", "boosts: none" — because to a small model an absent line is a
   vacuum to fill, not evidence of absence. Fixed ordering is also what
   keeps the rendering byte-stable for a given state, which the prompt
   cache depends on.

3. NEVER BREAK. This feeds a live consult path; any internal surprise
   degrades to a placeholder token, never an exception. The public entry
   point is wrapped whole as the last resort.

The sheet renders WORLD-0 of the determinized search (the caller passes
states[0]); the `assumed:` lines are world-0's sample specifically, which
is what lets the overlay judge that world's assumptions against the usage
priors already in the dossier. Normalized ids throughout — this is the
reasoning register. A caster-facing rendering would need display names
(measured: engine-speak makes the characters free-associate) and is a
separate, untested consumer.
"""

from __future__ import annotations

import re

_STAT_NAMES = ("atk", "def", "spa", "spd", "spe", "acc", "eva")
_EV_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")

# volatile-status durations the engine tracks turn counts for; everything
# else in volatile_statuses renders bare
_TIMED_VOLATILES = ("confusion", "encore", "lockedmove", "slowstart",
                    "taunt", "yawn")


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _pct(hp, maxhp) -> int:
    try:
        return round(100 * hp / maxhp) if maxhp else 0
    except Exception:
        return 0


def _boosts(side) -> str:
    """'atk+2 spe-1' or 'none'. Boosts live on the SIDE (they are the
    active mon's) and are protocol-derived on both sides — always fact."""
    out = []
    for name, attr in (("atk", "attack_boost"), ("def", "defense_boost"),
                       ("spa", "special_attack_boost"),
                       ("spd", "special_defense_boost"),
                       ("spe", "speed_boost"), ("acc", "accuracy_boost"),
                       ("eva", "evasion_boost")):
        v = getattr(side, attr, 0) or 0
        if v:
            out.append(f"{name}{v:+d}")
    return " ".join(out) or "none"


def _status(mon, side) -> str:
    """Status token with its counters attached — 'toxic(ctr=3)' reads as a
    clock where bare 'toxic' does not, and the counter lives on the SIDE's
    conditions rather than the mon. Lowercased first: a from_string round
    trip yields UPPERCASE tokens, and 'NONE' sailing past the == 'none'
    check printed a phantom status on every healthy mon."""
    st = _norm(getattr(mon, "status", "none")) or "none"
    if st == "none":
        return "none"
    bits = []
    if st in ("toxic", "tox"):
        n = getattr(getattr(side, "side_conditions", None), "toxic_count", 0)
        if n:
            bits.append(f"ctr={n}")
    if st in ("sleep", "slp"):
        rest = getattr(mon, "rest_turns", 0) or 0
        slept = getattr(mon, "sleep_turns", 0) or 0
        if rest:
            bits.append(f"rest={rest}")
        elif slept:
            bits.append(f"slept={slept}")
    return f"{st}({', '.join(bits)})" if bits else st


def _hazards(sc) -> str:
    return (f"SR={'yes' if getattr(sc, 'stealth_rock', 0) else 'no'} "
            f"spikes={getattr(sc, 'spikes', 0)} "
            f"tspikes={getattr(sc, 'toxic_spikes', 0)} "
            f"web={'yes' if getattr(sc, 'sticky_web', 0) else 'no'}")


def _screens(sc) -> str:
    bits = []
    for name, attr in (("reflect", "reflect"), ("lightscreen", "light_screen"),
                       ("veil", "aurora_veil")):
        v = getattr(sc, attr, 0) or 0
        if v:
            bits.append(f"{name}={v}t")
    return " ".join(bits) or "none"


def _extra_conditions(sc) -> list[str]:
    """Rarer side conditions render only when set (safeguard, mist, ...) —
    the fixed-slot rule covers the axes models actually hallucinate about;
    printing nineteen zeros a side would drown them."""
    out = []
    for name, attr in (("tailwind", "tailwind"), ("safeguard", "safeguard"),
                       ("mist", "mist"), ("luckychant", "lucky_chant"),
                       ("healingwish", "healing_wish"),
                       ("lunardance", "lunar_dance")):
        v = getattr(sc, attr, 0) or 0
        if v:
            out.append(f"{name}={v}t")
    return out


def _volatiles(side) -> str:
    vs = sorted(_norm(v) for v in (getattr(side, "volatile_statuses", None)
                                   or set()))
    dur = getattr(side, "volatile_status_durations", None)
    timed = {}
    for name in _TIMED_VOLATILES:
        v = getattr(dur, name, 0) or 0
        if v:
            timed[name] = v
    out = []
    for v in vs:
        out.append(f"{v}({timed[v]}t)" if v in timed else v)
    for name, v in timed.items():        # duration without the set entry
        if name not in vs:
            out.append(f"{name}({v}t)")
    sub = getattr(side, "substitute_health", 0) or 0
    if sub and "substitute" not in vs:
        out.append("substitute")
    line = " ".join(out) or "none"
    if sub:
        line += f" subhp={sub}"
    return line


def _last_move(side, mons) -> str | None:
    """'move:SUCKERPUNCH' -> 'suckerpunch'; 'switch:3' -> 'switch(gliscor)'
    by resolving the slot index — the raw index rendered as 'last_move=3',
    a number the model can only misread."""
    raw = str(getattr(side, "last_used_move", "") or "").lower()
    if raw.startswith("switch:"):
        rest = raw[7:]
        if rest.isdigit() and int(rest) < len(mons):
            return f"switch({_norm(getattr(mons[int(rest)], 'id', '?'))})"
        return None if not rest else f"switch({_norm(rest)})"
    rest = _norm(raw.removeprefix("move:"))
    if rest and rest != "none" and not rest.isdigit():
        return rest
    return None


def _side_state_lines(label, side, mons) -> list[str]:
    sc = getattr(side, "side_conditions", None)
    lines = [f"{label} hazards: {_hazards(sc)} | screens: {_screens(sc)}"]
    extra = _extra_conditions(sc)
    if extra:
        lines[0] += " | " + " ".join(extra)
    bits = [f"boosts: {_boosts(side)}", f"volatiles: {_volatiles(side)}"]
    wish = getattr(side, "wish", (0, 0)) or (0, 0)
    if wish[0]:
        bits.append(f"wish=incoming({wish[1]}hp)")
    fs = getattr(side, "future_sight", (0, "0")) or (0, "0")
    if fs[0]:
        bits.append(f"futuresight={fs[0]}t")
    if getattr(side, "force_trapped", False):
        bits.append("trapped=yes")
    last = _last_move(side, mons)
    if last:
        bits.append(f"last_move={last}")
    lines.append(f"{label} active-state: " + " | ".join(bits))
    return lines


def _tera_available(side) -> bool:
    return not any(getattr(p, "terastallized", False)
                   for p in getattr(side, "pokemon", None) or [])


def _moves_with_pp(mon) -> list[str]:
    out = []
    for mv in getattr(mon, "moves", None) or []:
        mid = _norm(getattr(mv, "id", "none"))
        if mid in ("", "none"):
            continue
        s = f"{mid} pp{getattr(mv, 'pp', 0)}"
        if getattr(mv, "disabled", False):
            s += "(disabled)"
        out.append(s)
    return out


def _mons_and_active(side):
    """(mons list, active index). Materialised ONCE: the pyo3 getters
    build fresh Python wrappers on every attribute access, so identity
    tests against a second `side.pokemon` read never match — iteration
    must be by index over a single snapshot."""
    mons = list(getattr(side, "pokemon", None) or [])
    try:
        idx = int(getattr(side, "active_index", 0))
    except Exception:
        idx = 0
    if not (0 <= idx < len(mons)):
        idx = -1
    return mons, idx


def _battle_index(mons_dict) -> dict:
    """_norm(species) -> poke-env mon, tolerant of None."""
    out = {}
    for m in (mons_dict or {}).values():
        out[_norm(getattr(m, "species", ""))] = m
    return out


def _revealed_moves(bmon) -> set:
    return {_norm(k) for k in (getattr(bmon, "moves", None) or {})}


def _item_revealed(bmon) -> str | None:
    it = getattr(bmon, "item", None)
    return it if it and it != "unknown_item" else None


def _our_mon_lines(mon, side, active: bool) -> list[str]:
    head = "active" if active else "bench"
    hp = getattr(mon, "hp", 0)
    maxhp = getattr(mon, "maxhp", 1)
    line = (f"US {head}: {_norm(getattr(mon, 'id', '?')) or '?'} "
            f"hp={hp}/{maxhp}({_pct(hp, maxhp)}%) "
            f"status={_status(mon, side)}")
    if getattr(mon, "terastallized", False):
        line += f" TERASTALLIZED={_norm(getattr(mon, 'tera_type', '?'))}"
    else:
        line += f" tera_type={_norm(getattr(mon, 'tera_type', '?'))}"
    out = [line,
           f"  item={_norm(getattr(mon, 'item', 'none')) or 'none'} "
           f"ability={_norm(getattr(mon, 'ability', 'none')) or 'none'} "
           f"moves: {' | '.join(_moves_with_pp(mon)) or 'none'}"]
    return out


def _their_mon_lines(mon, side, bmon, active: bool) -> list[str]:
    """The revealed/assumed split. `bmon` is the poke-env mon (protocol
    truth) or None — with no battle to diff against, NOTHING about the set
    is treated as revealed, which is the conservative direction."""
    head = "active" if active else "bench"
    hp = getattr(mon, "hp", 0)
    maxhp = getattr(mon, "maxhp", 1)
    line = (f"THEM {head}: {_norm(getattr(mon, 'id', '?')) or '?'} "
            f"hp={_pct(hp, maxhp)}% "
            f"status={_status(mon, side)}")
    if getattr(mon, "terastallized", False):
        line += f" TERASTALLIZED={_norm(getattr(mon, 'tera_type', '?'))}"
    out = [line]

    seen = _revealed_moves(bmon) if bmon is not None else set()
    revealed, assumed = [], []
    for mv in getattr(mon, "moves", None) or []:
        mid = _norm(getattr(mv, "id", "none"))
        if mid in ("", "none"):
            continue
        (revealed if mid in seen else assumed).append(mid)

    rev_bits = []
    if revealed:
        rev_bits.append("moves=" + ",".join(revealed))
    item_seen = _item_revealed(bmon) if bmon is not None else None
    if item_seen:
        rev_bits.append(f"item={_norm(item_seen)}")
    abil_seen = getattr(bmon, "ability", None) if bmon is not None else None
    if abil_seen:
        rev_bits.append(f"ability={_norm(abil_seen)}")
    tera_seen = getattr(bmon, "tera_type", None) if bmon is not None else None
    if tera_seen:
        rev_bits.append(f"tera={_norm(tera_seen)}")
    out.append("  revealed: " + (" | ".join(rev_bits) or "nothing yet"))

    asm_bits = []
    if assumed:
        asm_bits.append("moves+=" + ",".join(assumed))
    if not item_seen:
        asm_bits.append(f"item={_norm(getattr(mon, 'item', 'none')) or 'none'}")
    if not abil_seen:
        asm_bits.append(
            f"ability={_norm(getattr(mon, 'ability', 'none')) or 'none'}")
    if not tera_seen and not getattr(mon, "terastallized", False):
        asm_bits.append(f"tera={_norm(getattr(mon, 'tera_type', '?'))}")
    # The translator bakes spreads into RAW STATS and leaves evs/nature at
    # the constructor defaults (serious, 85s across) — rendering those
    # defaults as "spread=serious 85/85/..." would launder a placeholder
    # into a claim, the exact failure this module exists to prevent. Only
    # a spread that differs from the defaults carries information; the
    # modeled speed is always real and always worth a line.
    nat = _norm(getattr(mon, "nature", "")) or "serious"
    evs = tuple(getattr(mon, "evs", None) or ())
    if len(evs) == 6 and (nat != "serious" or set(evs) != {85}):
        asm_bits.append(f"spread={nat} {'/'.join(str(v) for v in evs)}")
    asm_bits.append(f"speed={getattr(mon, 'speed', '?')}")
    out.append("  assumed(world-0 sample): " + (" | ".join(asm_bits)
                                                or "nothing"))
    return out


def _fainted_names(mons_dict) -> list[str]:
    """From the BATTLE side: the engine replaces fainted mons with blank
    dummies that no longer know their species."""
    return sorted(_norm(getattr(m, "species", ""))
                  for m in (mons_dict or {}).values()
                  if getattr(m, "fainted", False))


def _field_lines(state) -> list[str]:
    w = _norm(getattr(state, "weather", "none")) or "none"
    wt = getattr(state, "weather_turns_remaining", 0) or 0
    t = _norm(getattr(state, "terrain", "none")) or "none"
    tt = getattr(state, "terrain_turns_remaining", 0) or 0
    tr = getattr(state, "trick_room", False)
    trt = getattr(state, "trick_room_turns_remaining", 0) or 0
    # weather/terrain turn counts hinge on unrevealed extender items
    # (Heat Rock and friends), so the count is the engine's estimate even
    # though the condition itself is protocol fact
    parts = [f"weather={w}" + (f"({wt}t est)" if w != "none" and wt >= 0 else ""),
             f"terrain={t}" + (f"({tt}t est)" if t != "none" else ""),
             f"trickroom={'yes(' + str(trt) + 't)' if tr else 'no'}"]
    return ["FIELD " + " | ".join(parts)]


def _beliefs_lines(obs, species: list[str]) -> list[str]:
    """Play-derived eliminations from the belief system, one register above
    a prior (they came from THIS game) but below a reveal. Optional: only
    rendered when the caller has a BattleObservations to give."""
    lines = ["BELIEFS (inferred from this game's play):"]
    found = False
    for sp in species:
        bits = []
        try:
            forb = obs.forbidden(sp)
        except Exception:
            forb = ()
        if forb:
            bits.append("not " + "/".join(sorted(forb)))
        try:
            boots = obs.boots_inferred(sp)
        except Exception:
            boots = None
        if boots:
            bits.append(str(boots))
        if bits:
            found = True
            lines.append(f"  {sp}: {'; '.join(bits)}")
    if not found:
        lines.append("  none recorded")
    return lines


def render_sheet(state, battle=None, obs=None, turn=None) -> str:
    """The full board, world-0, fixed slots, revealed/assumed split.

    `state` is a poke_engine.State (side_one = us, the translator's
    convention); `battle` is the poke-env battle whose reveals decide what
    counts as fact on their side (None -> their whole set is assumed);
    `obs` is an optional set_inference.BattleObservations. Never raises.
    """
    try:
        return _render(state, battle, obs, turn)
    except Exception as exc:
        return f"(board sheet unavailable: {exc.__class__.__name__})"


def _render(state, battle, obs, turn) -> str:
    if turn is None and battle is not None:
        turn = getattr(battle, "turn", None)
    lines = [f"=== BOARD turn={turn if turn is not None else '?'} "
             "(US exact; THEM split into revealed vs assumed) ==="]
    lines += _field_lines(state)

    s1 = getattr(state, "side_one", None)
    s2 = getattr(state, "side_two", None)
    ours_f = _fainted_names(getattr(battle, "team", None)) if battle else []
    theirs_f = (_fainted_names(getattr(battle, "opponent_team", None))
                if battle else [])

    def alive(side, fallback_fainted):
        mons = getattr(side, "pokemon", None) or []
        n = sum(1 for p in mons if getattr(p, "hp", 0) > 0)
        return n, fallback_fainted

    n1, _ = alive(s1, ours_f)
    n2, _ = alive(s2, theirs_f)
    lines.append(
        f"ALIVE us={n1}/6 (fainted: {', '.join(ours_f) or 'none'}) | "
        f"them={n2}/6 (fainted: {', '.join(theirs_f) or 'none'})")
    lines.append(
        f"TERA us={'available' if _tera_available(s1) else 'USED'} | "
        f"them={'available' if _tera_available(s2) else 'USED'}")

    # --- us ---
    lines.append("--- US (sets exact) ---")
    mons1, act1 = _mons_and_active(s1)
    lines += _side_state_lines("US", s1, mons1)
    if act1 >= 0 and getattr(mons1[act1], "hp", 0) > 0:
        lines += _our_mon_lines(mons1[act1], s1, active=True)
    for i, p in enumerate(mons1):
        if i == act1 or getattr(p, "hp", 0) <= 0:
            continue
        lines += _our_mon_lines(p, s1, active=False)

    # --- them ---
    lines.append("--- THEM (hp/status/boosts/faints are fact; "
                 "sets partly sampled) ---")
    mons2, act2 = _mons_and_active(s2)
    lines += _side_state_lines("THEM", s2, mons2)
    bindex = _battle_index(getattr(battle, "opponent_team", None)
                           if battle else None)
    if act2 >= 0 and getattr(mons2[act2], "hp", 0) > 0:
        lines += _their_mon_lines(mons2[act2], s2,
                                  bindex.get(_norm(mons2[act2].id)),
                                  active=True)
    for i, p in enumerate(mons2):
        if i == act2 or getattr(p, "hp", 0) <= 0:
            continue
        lines += _their_mon_lines(p, s2, bindex.get(_norm(p.id)),
                                  active=False)

    if obs is not None:
        species = [_norm(p.id) for p in getattr(s2, "pokemon", None) or []
                   if getattr(p, "hp", 0) > 0]
        lines += _beliefs_lines(obs, species)
    return "\n".join(lines)
