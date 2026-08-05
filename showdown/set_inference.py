# In-battle opponent set refinement from observations.
#
# The loss-trace analysis (150 foul-play A/B games) showed the core defect of
# static set priors: the real scarf Iron Valiant was modeled as Booster
# Energy (80% chaos prior) all game, every game — our search kept "winning"
# positions that a faster-than-modeled sweeper then deleted (0.97 -> 0.05
# eval cliffs). Foul-play refines sets from what it observes; this module
# does the same from poke-env's retained protocol history:
#
#   - SPEED FLOORS: if the opponent's active moved before ours at equal
#     priority (no trick room, no speed boosts on their side, both moves
#     damaging), its effective speed exceeds ours at that moment. When the
#     modeled stat contradicts a floor, upgrade to Choice Scarf, then to a
#     max-speed spread if scarf alone isn't enough.
#   - SPEED CEILINGS: symmetrically, if they moved AFTER us when the model
#     says they're faster, drop an inferred scarf, then clamp the raw stat —
#     which also captures slower spreads and rare speed-drop items (Iron
#     Ball, Macho Brace) without guessing which one it is. Floors win over
#     ceilings when observations conflict.
#   - DAMAGE BRACKETS: a non-crit, boost-free, non-tera hit that exceeds a
#     MAX-INVESTED attacker's maximum roll by >15% proves a boosting item.
#     The WEAKEST item that explains the hit is chosen: Life Orb (1.3x,
#     boosts both categories) up to ~1.38x, Choice Band/Specs beyond. The
#     denominator is max investment, NOT our canonical-spread guess: an
#     attacker invests in Atk or SpA as a matter of course, and billing that
#     to an item branded 28-99% of sets Choice-locked for free.
#
# Observations only ever apply to inferred details — revealed items are
# never overridden.

from __future__ import annotations

import os

import poke_engine as pe

# Measure damage-bracket ratios against a max-invested attacker rather than
# our canonical-spread guess. Default ON: it removes a false certainty rather
# than sharpening an estimate. "0" restores the old denominator.
_INVESTED_DENOM = os.environ.get("CB_DAMAGE_INVESTED", "1") != "0"

_DAMAGING = ("Physical", "Special")

_gen9_moves = None
_gen9_dex = None


def _moves_data():
    global _gen9_moves
    if _gen9_moves is None:
        from poke_env.data.gen_data import GenData
        _gen9_moves = GenData.from_gen(9).moves
    return _gen9_moves


def _dex_data():
    global _gen9_dex
    if _gen9_dex is None:
        from poke_env.data.gen_data import GenData
        _gen9_dex = GenData.from_gen(9).pokedex
    return _gen9_dex


def _abilities_of(species_norm: str) -> set[str]:
    return {_normalize(str(a)) for a
            in _dex_data().get(species_norm, {}).get("abilities", {}).values()}


def _can_magic_guard(species_norm: str) -> bool:
    """Magic Guard also nullifies hazard chip, so it's the confound for the
    Boots negative-evidence read: if the species can run it, zero-chip entry
    doesn't prove Boots."""
    return "magicguard" in _abilities_of(species_norm)


# NEGATIVE ability evidence (2026-08-04, from the fp cross-audit): these
# abilities ANNOUNCE THEMSELVES on switch-in, so a first entry that stays
# silent rules them out for the species. Only unconditional announcers
# qualify for the flat set; weather/terrain setters announce only when they
# would CHANGE the field, so they carry their expected result and are
# checked against the field state at entry. Deliberately absent:
# supremeoverlord (silent at fallen=0), frisk (silent vs itemless),
# screencleaner (silent without screens), trace (can fail on untraceable),
# protosynthesis/quarkdrive (condition on sun/Booster). Dauntless Shield
# and Intrepid Sword announce only ONCE per battle in gen9 — safe here
# because only a species' FIRST entry ever opens the latch.
_ANNOUNCE_ALWAYS = frozenset({
    "intimidate", "pressure", "unnerve", "download", "dauntlessshield",
    "intrepidsword", "vesselofruin", "swordofruin", "tabletsofruin",
    "beadsofruin", "slowstart", "moldbreaker", "teravolt", "turboblaze",
})
_ANNOUNCE_WEATHER = {"drought": "sun", "orichalcumpulse": "sun",
                     "drizzle": "rain", "sandstream": "sand",
                     "snowwarning": "snow"}
_ANNOUNCE_TERRAIN = {"electricsurge": "electric", "hadronengine": "electric",
                     "grassysurge": "grassy", "mistysurge": "misty",
                     "psychicsurge": "psychic"}

# Outrage-class rampages: once clicked, the user has no choice until the
# rampage ends (2-3 turns, then fatigue confusion). The protocol marks
# neither the lock nor its end — both are derived from consecutive use.
_LOCK_MOVES = frozenset({"outrage", "petaldance", "thrash", "ragingfury"})

# ON-HIT negative ability evidence (user idea, 2026-08-05): the switch-in
# announce rule's sibling. Absorb/immunity abilities react DETERMINISTICALLY
# when hit by their trigger — so our damaging move CONNECTING FOR DAMAGE
# proves the reactive ability absent (had they run it, the -immune/-heal/
# boost announce would have replaced the damage line). Deterministic
# reactors only; proc-chance abilities (Static, Flame Body) prove nothing
# by staying quiet, and Multiscale is silent by design (damage-calc class,
# not announce class). Marquee case: one Water hit on a Clodsire proves
# not-waterabsorb, i.e. Unaware.
_TYPE_REACTIVE = {
    "electric": ("voltabsorb", "lightningrod", "motordrive"),
    "water": ("waterabsorb", "stormdrain", "dryskin"),
    "grass": ("sapsipper",),
    "ground": ("levitate", "eartheater"),
    "fire": ("flashfire", "wellbakedbody"),
}
_FLAG_REACTIVE = {"sound": ("soundproof",), "bullet": ("bulletproof",),
                  "wind": ("windrider",)}
_PRIORITY_REACTIVE = ("dazzling", "queenlymajesty", "armortail")
# attackers that pierce abilities void the evidence (our side, always known)
_ABILITY_IGNORERS = frozenset({"moldbreaker", "teravolt", "turboblaze"})

# ON-HIT item evidence (user idea, 2026-08-05): items that MUST announce
# when their trigger lands. Unlike abilities the announce (eat/boost/
# enditem) FOLLOWS the damage line, so verdicts are latched at the hit and
# convicted at the next commitment point unless an announce cleared them.
# Focus Sash needs no latch: a KO from full HP is itself the proof.
_ONHIT_TYPE_ITEMS = {"water": ("absorbbulb", "luminousmoss"),
                     "electric": ("cellbattery",), "ice": ("snowball",)}
_PINCH_50 = ("sitrusberry",)
_PINCH_25 = ("figyberry", "wikiberry", "magoberry", "aguavberry",
             "iapapaberry")
# weakness (SE-reducing) berries, keyed by attacking type; they announce
# when they halve a super-effective hit, so an SE hit that connects
# unannounced rules the matching berry out. Chilan is the special case:
# Normal is never super-effective, so it triggers on ANY Normal hit.
_WEAKNESS_BERRIES = {
    "fire": "occaberry", "water": "passhoberry", "electric": "wacanberry",
    "grass": "rindoberry", "ice": "yacheberry", "fighting": "chopleberry",
    "poison": "kebiaberry", "ground": "shucaberry", "flying": "cobaberry",
    "psychic": "payapaberry", "bug": "tangaberry", "rock": "chartiberry",
    "ghost": "kasibberry", "dragon": "habanberry", "dark": "colburberry",
    "steel": "babiriberry", "fairy": "roseliberry",
}

# Two-turn charge moves, named exactly as the engine's charging volatiles
# (poke-engine charge_choice_to_volatile): mid-charge the user is committed
# and often semi-invulnerable, announced only by |-prepare|.
_CHARGE_MOVES = frozenset({
    "bounce", "dig", "dive", "fly", "freezeshock", "geomancy", "iceburn",
    "meteorbeam", "electroshot", "phantomforce", "razorwind", "shadowforce",
    "skullbash", "skyattack", "skydrop", "solarbeam", "solarblade",
})


def _grounded_by_species(species_norm: str) -> bool:
    """True when the species is grounded on type + ability alone — not a
    Flying type and cannot plausibly run Levitate. Spikes only chip grounded
    Pokemon, so a grounded mon that avoids Spikes is Boots evidence; a
    Flying/maybe-Levitate one avoids them legitimately (no signal). Air
    Balloon is the remaining airborne source and is caught at run time (it
    announces itself on switch-in)."""
    entry = _dex_data().get(species_norm, {})
    types = {str(t).lower() for t in entry.get("types", [])}
    if "flying" in types:
        return False
    return "levitate" not in _abilities_of(species_norm)


def _normalize(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _move_info(move_name: str) -> tuple[int, str]:
    """(priority, category) with safe defaults."""
    entry = _moves_data().get(_normalize(move_name), {})
    return entry.get("priority", 0), entry.get("category", "Status")


def _move_type(move_name: str) -> str:
    return (_moves_data().get(_normalize(move_name), {})
            .get("type", "Normal")).lower()


# Moves whose damage cannot be measured against our synthetic probe state.
#
# A damage observation is re-evaluated by rebuilding a two-mon State and
# calling calculate_damage. That state reproduces stats, types, abilities,
# items and the weather — and NOTHING ELSE. It does not reproduce turn
# order, terrain, either side's status, the defender's HP at the moment of
# the hit, hit counts, boosts outside Atk/SpA, or how many mons have
# fainted. When a move's damage depends on any of those, the modelling error
# does not vanish: it lands in the ratio, and the ratio's only output is an
# ITEM CLAIM.
#
# Found 2026-08-03 from a Ting-Lu branded Choice Specs whose only special
# move is RUINATION — percentage damage, which poke-engine models correctly
# but against the defender's CURRENT HP. Evidence is collected across turns
# and re-evaluated later, so as our mon takes chip the modeled damage falls
# while the recorded hit stays fixed: the same observation reads 1.00 at
# full HP, 1.61 at 200/323, and 4.03 at 80/323. No item involved anywhere.
#
# Two rules are data-driven and cover most of it:
#   basePower 0  fixed, percentage, counter and variable-power moves
#                (Ruination, Super Fang, Seismic Toss, Night Shade, Endeavor,
#                Counter/Mirror Coat/Metal Burst, Flail, Reversal, Gyro Ball,
#                Electro Ball, Punishment, Final Gambit, the OHKOs)
#   multihit     calculate_damage returns ONE hit's rolls while the protocol
#                reports the TOTAL, so a 5-hit Rock Blast reads ~4x
# The set below is the rest: real base power, but power conditional on state
# the probe does not carry. Facade is the headline — modeled unstatused it
# reads 254, but a Guts user with a Flame Orb actually hits for 754, a 2.97x
# ratio that brands Choice Band every single time.
_UNMODELED_DAMAGE = frozenset({
    # attacker or target status
    "facade", "hex", "venoshock", "barbbarrage", "infernalparade",
    # defender's current HP
    "brine", "hardpress", "wringout", "crushgrip",
    # attacker's current HP
    "eruption", "waterspout", "dragonenergy",
    # turn order / prior damage this turn. Sucker Punch and Steel Roller are
    # deliberately NOT here: their condition makes them FAIL, it does not
    # change their power (70 and 130 flat), so a hit that landed is
    # measurable like any other.
    "boltbeak", "fishiousrend", "payback", "avalanche", "assurance",
    "revenge",
    # terrain (never set on the probe state)
    "risingvoltage", "expandingforce", "psyblade", "mistyexplosion",
    "terrainpulse",
    # Weather Ball alone: the probe DOES carry the recorded weather, so its
    # power is reproducible — but its TYPE changes with weather while
    # _move_type reports the dex value (Normal), which would drop it in the
    # wrong bucket for the type-item branch. Solar Beam and Solar Blade keep
    # their type and track weather correctly (183 clear, 93 in rain), so
    # they stay measurable.
    "weatherball",
    # running counters: boosts outside Atk/SpA, hits taken, faints, repeats
    "storedpower", "powertrip", "ragefist", "lastrespects", "echoedvoice",
    "furycutter", "rollout", "iceball", "spitup", "trumpcard",
    # item-derived power — circular here, since the item is what we infer
    "acrobatics", "fling", "naturalgift",
    # tera changes the type and the STAB math
    "terablast", "terastarstorm",
    # RANDOM multiplier — not a function of any state, so no probe could
    # reproduce it. Fickle Beam doubles on a 30% roll (80 -> 160 BP), which
    # is a 2x ratio landing squarely in the Choice bracket roughly one hit
    # in three. The protocol does announce the proc, but we do not parse it;
    # if that changes, this one can move to the measurable side. The other
    # random-power moves (Magnitude, Present, Psywave, Beat Up) are all base
    # power 0 and already fall to the data-driven rule.
    "ficklebeam",
})


# Base power 0 in the dex, but the damage IS a deterministic function of
# inputs the probe carries exactly: both sides' weight_kg, straight from the
# dex. Verified against the engine — Grass Knot reads 33/123/183 against a
# 5/50/200kg target, Heavy Slam the inverse. Autotomize and Float Stone
# halve the attacker's weight without our tracking it, which makes the
# modeled hit too STRONG and the ratio too LOW: it can cost us an item we
# would otherwise infer, never invent one.
_WEIGHT_BASED = frozenset({"grassknot", "lowkick", "heavyslam", "heatcrash"})

# Damaging moves carried for their EFFECT, not for the stat behind them: a
# pivot, a hazard, a removal, a knock. They must not count toward "this mon
# attacks with both categories", because a special attacker running U-turn is
# the single most ordinary set in the tier — Pelipper's Choice Specs build is
# Hurricane / Hydro Pump / U-turn / Roost, and it produced the one CORRECT
# Choice brand in the sample. A naive category test would have killed it.
_UTILITY_DAMAGING = frozenset({
    "uturn", "flipturn", "voltswitch", "knockoff", "rapidspin", "mortalspin",
    "ceaselessedge", "stoneaxe",
})


def _offensive_categories(move_ids) -> set[str]:
    """Categories this mon actually ATTACKS with, utility moves removed."""
    cats: set[str] = set()
    for move_id in move_ids:
        mid = _normalize(move_id)
        if mid in _UTILITY_DAMAGING:
            continue
        entry = _moves_data().get(mid) or {}
        if entry.get("category") in _DAMAGING and (
                entry.get("basePower") or mid in _WEIGHT_BASED):
            cats.add(entry["category"])
    return cats


def _measurable_damage(move_id: str) -> bool:
    """Can this move's damage be reproduced by the synthetic probe state?

    Conservative by design: an unmeasurable observation is not weak evidence,
    it is evidence of the wrong thing, and the only claim it can produce is
    a false lock. Dropping a real Life Orb hit costs us a slightly soft
    damage model; keeping a false Choice brand tells the search the target is
    locked into one move it may in fact switch off freely.
    """
    mid = _normalize(move_id)
    entry = _moves_data().get(mid)
    if not entry:
        return False
    if entry.get("multihit"):
        return False
    if not entry.get("basePower") and mid not in _WEIGHT_BASED:
        return False
    return mid not in _UNMODELED_DAMAGE


# type-boosting held items (1.2x one type). Inferred only when boosted hits
# are confined to one type while another damaging type reads clean.
_TYPE_ITEM = {
    "fire": "charcoal", "water": "mysticwater", "electric": "magnet",
    "grass": "miracleseed", "ice": "nevermeltice", "fighting": "blackbelt",
    "poison": "poisonbarb", "ground": "softsand", "flying": "sharpbeak",
    "psychic": "twistedspoon", "bug": "silverpowder", "rock": "hardstone",
    "ghost": "spelltag", "dragon": "dragonfang", "dark": "blackglasses",
    "steel": "metalcoat", "fairy": "fairyfeather",
}


class BattleObservations:
    """Incremental scanner over battle._replay_data producing set evidence."""

    def __init__(self):
        self._cursor = 0
        # opp species -> required raw (stat x item-multiplier) lower bound
        self.speed_floor: dict[str, float] = {}
        # opp species -> raw (stat x item-multiplier) upper bound
        self.speed_ceiling: dict[str, float] = {}
        # dicts: species/move/damage/our_species/weather
        self.damage_evidence: list[dict] = []
        # opp species -> the INFERRED item the translator has adopted for it
        # (choicescarf / lifeorb / choiceband / ...). Written by the
        # translator's build loop when an observation upgrades an unrevealed
        # item; read by the live player to emit "set reveal" commentary
        # beats the moment a belief is confirmed. Never holds revealed items.
        self.confirmed: dict[str, str] = {}

        # negative-evidence Heavy-Duty Boots: a mon switched in over our
        # Stealth Rock and took ZERO chip. Nothing else prevents SR damage
        # on entry except Boots or Magic Guard, so a species that cannot run
        # Magic Guard is confidently Boots; a Magic-Guard-capable one is
        # ambiguous (recorded, not promoted — the search must not model the
        # wrong item, though a caster may hedge on it).
        self.boots: set[str] = set()            # confident Heavy-Duty Boots
        self.boots_ambiguous: set[str] = set()  # zero-chip but MG-capable

        # Regenerator proven behaviorally: a mon that left the field alive
        # and returned with MORE HP was healed on the bench, and Regenerator
        # is the only passive bench heal in gen9 (Healing Wish / Revival
        # Blessing announce -heal events, and the switch line shows the
        # PRE-heal hp, so they can't fake this). Dex-gated at record time.
        self.regen: set[str] = set()

        # negative ability evidence: entry announcers that stayed silent on
        # the species' FIRST switch-in (see _ANNOUNCE_ALWAYS)
        self.impossible_abilities: dict[str, set] = {}
        self._ability_latch: dict | None = None  # open first-entry watch
        self._entered: set[str] = set()          # opp species seen entering
        self._nogas = False   # Neutralizing Gas seen -> announce rule is off
        self._terrain = "none"                   # for the Surge-class check

        # substitute bookkeeping: per sub-OWNER role, every hit the standing
        # sub has absorbed as (attacker_species, move_id). The protocol
        # never states sub damage (that concealment is the move's point), so
        # remaining HP is ESTIMATED downstream from these; multihit moves
        # emit one [damage] activation per hit, which pairs exactly with
        # calculate_damage returning one hit's rolls.
        self.sub_hits: dict[str, list] = {"p1": [], "p2": []}
        self._last_move: dict[str, tuple[str, str]] = {}  # role -> (sp, mid)
        # Outrage-class rampage: role -> (species, move_id, consecutive uses)
        self._rampage: dict[str, tuple | None] = {"p1": None, "p2": None}
        # two-turn charge in progress: role -> move id (from |-prepare|)
        self._charging: dict[str, str | None] = {"p1": None, "p2": None}
        self._opp_role: str | None = None
        # our just-used damaging move, awaiting its on-hit verdict
        self._our_attack: str | None = None
        self._our_se = False                       # this hit was super-eff.
        self._our_abilities: dict[str, str] = {}   # our species -> ability
        # on-hit ITEM verdicts latched at the damage line, convicted at the
        # next commitment point unless an announce named the item first
        self._hit_latch: dict | None = None

        # DIRECT reveals the protocol broadcasts in [from]/[of] tags — "was
        # poisoned by Toxic Orb", Flame Orb burns, Leftovers heals, Rocky
        # Helmet chip, ability activations. Not inference: the sim names the
        # source; we were dropping the tag (the audited Gliscor game had
        # item: null at T16 with the orb announced on the wire since T2).
        # opp species -> item id / ability id
        self.revealed_item: dict[str, str] = {}
        self.revealed_ability: dict[str, str] = {}

        # behavioral Choice disproof: two DISTINCT moves in one stint means
        # the mon is not currently Choice-locked — airtight, and it clears
        # wrongly-branded pure attackers the status-move veto can't reach
        # (measured 2026-08-02: 33 game-mons had an assumed Choice item
        # falsified by a later reveal). Called moves ([from] tags: Sleep
        # Talk, Dancer, locked-move continuations) and Struggle don't count.
        self.choice_disproven: set[str] = set()
        self._stint_moves: dict[str, set] = {}   # role -> move ids this stint

        # UNIFIED CONSTRAINT LAYER (2026-08-02). Before this, eliminations
        # lived in four separate places (`boots`, `speed_floor`, `confirmed`,
        # `choice_disproven`), each consulted by SOME tiers and not others —
        # which is how the Choice Band Gliscor survived: the damage-bracket
        # wrote `confirmed` and no later evidence re-checked it. This is one
        # per-species set of items the mon PROVABLY does not hold, and every
        # tier filters through it. Eliminations are DEDUCTIVE and therefore
        # safe to apply blindly; positive assertions (Boots from zero chip)
        # stay separate because they are abductive and need a uniqueness
        # argument (Magic Guard explains the same observation).
        self.impossible_items: dict[str, set] = {}
        # every damaging move each opposing species has been seen using,
        # across all stints — the mixed-attacker test needs the whole game
        self._seen_moves: dict[str, set] = {}
        # per-species HP state at the last end-of-turn, for upkeep proofs
        self._opp_hp: dict[str, tuple[int, int]] = {}
        self._healed_this_turn: set[str] = set()
        self._opp_unstatused: set[str] = set()
        # per-turn: opponent species that attacked, that we hit with a
        # CONTACT move, and the item announcements actually seen this turn
        self._opp_attacked: set[str] = set()
        self._we_contacted: set[str] = set()
        self._item_seen: set[tuple] = set()
        self._balloon_pending: set[str] = set()
        self._side_sr = {"p1": False, "p2": False}      # Stealth Rock per side
        self._side_spikes = {"p1": False, "p2": False}  # Spikes per side
        self._gravity = False                    # grounds everyone for Spikes
        self._entry_latch: dict | None = None    # open switch-in over hazards

        # ---- scanner state ----
        self._active: dict[str, str] = {}          # role -> species
        self._hp: dict[str, int] = {}              # our species -> current hp
        self._spe_boost = {"p1": 0, "p2": 0}
        self._atk_boost = {"p1": 0, "p2": 0}
        self._spa_boost = {"p1": 0, "p2": 0}
        self._par: set[str] = set()                # "role species" paralyzed
        self._tera: set[str] = set()               # roles that terastallized
        self._tailwind: set[str] = set()
        self._trick_room = False
        self._weather = "none"
        self._turn_moves: list[tuple[str, str]] = []
        self._pending: dict | None = None          # open attack on us

    # ---- helpers ----

    @staticmethod
    def _boost_mult(stages: int) -> float:
        return (2 + stages) / 2 if stages >= 0 else 2 / (2 - stages)

    def _forbid(self, species: str, *items: str):
        """Record items this species PROVABLY does not hold."""
        if not species:
            return
        self.impossible_items.setdefault(species, set()).update(items)

    def _event_species(self, event) -> str | None:
        """Details-consistent species for a slot-addressed event. Idents
        carry the NICKNAME, which for formes is the base name ('p2a:
        Slowking' for Slowking-Galar), so parsing the ident produced keys
        that never matched the details-derived ones the rest of the scanner
        uses — every upkeep proof and par-speed correction silently missed
        forme mons until the Regenerator test caught it (2026-08-04).
        Slot events are about the ACTIVE mon, so resolve through the active
        table; bench-addressed idents ('p2: Name', no slot letter — e.g.
        Revival Blessing heals) return None."""
        ident = str(event[2])
        pos = ident.split(":", 1)[0]
        if len(pos) < 3:
            return None
        return self._active.get(pos[:2])

    def forbidden(self, species: str) -> frozenset:
        """Items ruled out for this species. Every set tier filters on this."""
        return frozenset(self.impossible_items.get(_normalize(species), ()))

    def ability_forbidden(self, species: str) -> frozenset:
        """Abilities ruled out by a silent first switch-in."""
        return frozenset(self.impossible_abilities.get(_normalize(species),
                                                       ()))

    def rampage_for(self, species: str):
        """(species, move_id, consecutive_uses) when the OPPONENT'S active
        is mid-Outrage-class-rampage and matches, else None."""
        if self._opp_role is None:
            return None
        r = self._rampage.get(self._opp_role)
        return r if r is not None and r[0] == _normalize(species) else None

    def opp_sub_hits(self) -> list:
        """Hits absorbed by the opponent's standing substitute."""
        return list(self.sub_hits.get(self._opp_role or "", ()))

    def our_sub_hit(self) -> bool:
        """Whether OUR standing substitute has absorbed at least one hit."""
        our = "p1" if self._opp_role == "p2" else "p2"
        return bool(self.sub_hits.get(our))

    def charging(self, role: str) -> str | None:
        """Move id this role's active is mid-charge on, else None."""
        return self._charging.get(role)

    def opp_charging(self) -> str | None:
        """Move id the OPPONENT's active is mid-charge on, else None."""
        return self._charging.get(self._opp_role or "")

    def opp_moved_this_turn(self) -> bool:
        """Whether the opponent has already acted in the CURRENT turn — the
        fast/slow pivot discriminator for a mid-turn force switch."""
        return any(r == self._opp_role for r, _ in self._turn_moves)

    def has_damage_evidence(self, species: str) -> bool:
        sp = _normalize(species)
        return any(ev["species"] == sp for ev in self.damage_evidence)

    def spread_ruled_out(self, opp_probe: pe.Pokemon, our_mons: dict,
                         max_checks: int = 3) -> bool:
        """Reverse damage calc at the SPREAD level (fp parity, upper
        violations only): a candidate build whose maximum roll cannot reach
        an observed hit is not the build in front of us — drop the whole
        candidate, spread and all, not just the item.

        `opp_probe` is the CANDIDATE's build (its spread/item/ability), so
        the item multiplier is part of what's being tested — this is what
        separates max-Attack from bulky spreads off a single hit. UPPER
        violations only, deliberately: everything ungated by the evidence
        curation (screens, Multiscale, attacker burn) only DEFLATES damage,
        so an under-max hit never convicts, while an over-max hit is safe
        to. Tolerance absorbs HP rounding and the last EV point."""
        sp = _normalize(getattr(opp_probe, "id", "") or "")
        checked = 0
        for ev in reversed(self.damage_evidence):
            if checked >= max_checks:
                break
            if ev["species"] != sp:
                continue
            our = our_mons.get(ev["our_species"])
            if our is None:
                continue
            try:
                state = pe.State(
                    side_one=pe.Side(pokemon=[opp_probe] + [
                        pe.Pokemon.create_fainted() for _ in range(5)]),
                    side_two=pe.Side(pokemon=[our] + [
                        pe.Pokemon.create_fainted() for _ in range(5)]),
                    weather={"sun": pe.Weather.SUN, "rain": pe.Weather.RAIN,
                             "sand": pe.Weather.SAND, "snow": pe.Weather.SNOW,
                             "hail": pe.Weather.HAIL}.get(ev["weather"],
                                                          pe.Weather.NONE),
                    weather_turns_remaining=3 if ev["weather"] != "none"
                    else 0,
                )
                rolls = pe.calculate_damage(state, ev["move"], "splash",
                                            True)[0]
            except Exception:
                continue
            if not rolls or max(rolls) <= 0:
                continue
            checked += 1
            if ev["damage"] > max(rolls) + max(2, int(max(rolls) * 0.03)):
                return True
        return False

    def _close_turn_upkeep(self):
        """End-of-turn proofs from what did NOT happen.

        An item that heals or triggers at upkeep announces itself when it
        does. So a mon sitting below max HP through an end-of-turn WITHOUT a
        heal cannot be holding Leftovers or Black Sludge, and one that ends a
        turn unstatused cannot be holding Flame Orb or Toxic Orb (they self-
        inflict on the first upkeep they are held through).

        This is the elimination that actually pays: 5.0% of our assumptions
        about such mons still said Leftovers (measured 2026-08-02, 6,339
        assumptions), against 0.2% for the equivalent Boots case — because
        Leftovers is the most common item in the tier at 16.1% prior, so the
        prior puts real mass on exactly what the observation refutes.
        """
        for sp, (hp, maxhp) in self._opp_hp.items():
            if sp in self._healed_this_turn or hp <= 0:
                continue
            if hp < maxhp:
                self._forbid(sp, "leftovers", "blacksludge")
            if sp in self._opp_unstatused:
                self._forbid(sp, "flameorb", "toxicorb")

        # LIFE ORB: it costs the holder 1/10 max HP on every damaging move
        # and announces that recoil. A damaging move with no Life Orb damage
        # therefore proves no Life Orb — unless the species can run Sheer
        # Force (which cancels the recoil on secondary-effect moves) or Magic
        # Guard (which cancels it outright), the same confound shape as the
        # Boots read. 694 Life Orb announcements in our own logs.
        for sp in self._opp_attacked:
            if ("lifeorb", sp) in self._item_seen:
                continue
            if _abilities_of(sp) & {"sheerforce", "magicguard"}:
                continue
            self._forbid(sp, "lifeorb")

        # ROCKY HELMET: chips anything that makes CONTACT with the holder and
        # announces it. We hit them with a contact move and took no helmet
        # chip -> they are not holding one. 956 announcements logged, and at
        # 6.3% prior this is the largest of the three.
        for sp in self._we_contacted:
            if ("rockyhelmet", sp) not in self._item_seen:
                self._forbid(sp, "rockyhelmet")

        # AIR BALLOON announces itself on EVERY switch-in (532 in our logs),
        # so a switch-in with no announcement is proof of absence.
        for sp in self._balloon_pending:
            self._forbid(sp, "airballoon")

        self._healed_this_turn = set()
        self._opp_attacked = set()
        self._we_contacted = set()
        self._item_seen = set()
        self._balloon_pending = set()

    def _note_item_event(self, event, opp_role):
        """Record that an item ANNOUNCED itself this turn, for the
        absence-proofs in _close_turn_upkeep."""
        owner = None
        for a in event[3:]:
            if isinstance(a, str) and a.startswith("[of] ") and ":" in a:
                owner = a[5:].strip()[:2]
        role = owner or str(event[2])[:2]
        if role != opp_role:
            return
        sp = self._active.get(role)
        if not sp:
            return
        for a in list(event[3:]) + [event[3] if len(event) > 3 else ""]:
            if isinstance(a, str) and "item:" in a:
                self._item_seen.add((_normalize(a.split("item:")[1]), sp))
        if str(event[1]) == "-item" and len(event) > 3:
            name = _normalize(str(event[3]))
            self._item_seen.add((name, sp))
            if name == "airballoon":
                self._balloon_pending.discard(sp)
        if str(event[1]) == "-enditem" and len(event) > 3:
            # a consumed/removed item announced itself (berry eats, sash,
            # Weakness Policy, Eject Button...) — the hit latch reads this
            self._item_seen.add((_normalize(str(event[3])), sp))

    def _capture_reveals(self, event, opp_role):
        """Record [from] item:/ability: tags. An [of] tag reassigns the
        source (Rocky Helmet chip names the DEFENDER's item); otherwise the
        source is the mon the event is about. Opponent side only — our own
        set is known."""
        tags = [a for a in event[3:]
                if isinstance(a, str) and a.startswith("[")]
        if not tags:
            return
        of_role = None
        for a in tags:
            if a.startswith("[of] ") and ":" in a:
                of_role = a[5:].strip()[:2]
        for a in tags:
            for prefix, store in (("[from] item: ", self.revealed_item),
                                  ("[from] ability: ",
                                   self.revealed_ability)):
                if not a.startswith(prefix):
                    continue
                owner = of_role or event[2][:2]
                species = self._active.get(owner)
                if owner == opp_role and species:
                    store[species] = _normalize(a[len(prefix):])

    def _close_pending(self):
        p = self._pending
        self._pending = None
        if p is None or p["damage"] <= 0 or p["invalid"]:
            return
        # the observation must be one the probe state can reproduce at all
        if not _measurable_damage(p["move"]):
            return
        self.damage_evidence.append({
            "species": p["attacker"], "move": p["move"],
            "damage": p["damage"], "our_species": p["target"],
            "weather": p["weather"], "se": p["se"],
        })

    def _resolve_hit_latch(self):
        """Convict the latched on-hit item candidates: whatever DIDN'T
        announce by the next commitment point isn't held. An announced item
        clears itself (and, being the mon's one item slot, convicts every
        other candidate all the same). MUST run before _close_turn_upkeep
        at turn boundaries — upkeep clears _item_seen."""
        latch = self._hit_latch
        self._hit_latch = None
        if latch is None:
            return
        sp = latch["species"]
        announced = {item for (item, s) in self._item_seen if s == sp}
        self._forbid(sp, *(latch["items"] - announced))

    def _note_ability_announce(self, event, opp_role):
        """Record entry-ability announces while a first-entry watch is open:
        plain |-ability| lines and [from] ability: tags ([of] reassigns the
        owner, as in _capture_reveals)."""
        latch = self._ability_latch
        if (str(event[1]) == "-ability" and len(event) >= 4
                and str(event[2])[:2] == opp_role):
            latch["announced"].add(_normalize(str(event[3])))
            return
        of_role = None
        for a in event[3:]:
            if isinstance(a, str) and a.startswith("[of] ") and ":" in a:
                of_role = a[5:].strip()[:2]
        for a in event[3:]:
            if isinstance(a, str) and a.startswith("[from] ability: "):
                owner = of_role or str(event[2])[:2]
                if owner == opp_role:
                    latch["announced"].add(
                        _normalize(a[len("[from] ability: "):]))

    def _resolve_entry(self):
        """A switch-in over hazards closes its window (the next turn/move/
        switch after the immediate entry damage). Zero chip is Boots — or
        Magic Guard if the species can run it. Evidence is airtight for
        Stealth Rock (hits everything); for Spikes it only counts if the mon
        wasn't revealed airborne by an Air Balloon on entry (the species is
        already known grounded — that's the gate to latch on Spikes).

        Also closes the first-entry ABILITY watch: announce-class abilities
        the species could run that stayed silent are ruled out — weather and
        terrain setters only when they would have CHANGED the field."""
        alatch = self._ability_latch
        self._ability_latch = None
        if alatch is not None and not self._nogas:
            sp = alatch["species"]
            dex = _abilities_of(sp)
            expected = set(_ANNOUNCE_ALWAYS & dex)
            for ab, wx in _ANNOUNCE_WEATHER.items():
                if ab in dex and alatch["weather"] != wx:
                    expected.add(ab)
            for ab, tx in _ANNOUNCE_TERRAIN.items():
                if ab in dex and alatch["terrain"] != tx:
                    expected.add(ab)
            missing = expected - alatch["announced"]
            if missing:
                self.impossible_abilities.setdefault(sp, set()).update(
                    missing)

        latch = self._entry_latch
        self._entry_latch = None
        if latch is None or latch["chipped"]:
            return
        # SR proves it outright; Spikes proves it unless an Air Balloon made
        # the mon airborne — but under Gravity even a balloon is grounded, so
        # the balloon voids Spikes evidence only outside Gravity
        spikes_proven = latch["expects_spikes"] and (
            latch["gravity"] or not latch["airborne"])
        proven = latch["expects_sr"] or spikes_proven
        if not proven:
            return
        species = latch["species"]
        if _can_magic_guard(species):
            self.boots_ambiguous.add(species)
        else:
            self.boots.add(species)

    def _eval_turn_order(self, battle):
        """Both sides moved this turn: extract a speed bound if clean."""
        if len(self._turn_moves) != 2 or self._trick_room:
            return
        (r1, m1), (r2, m2) = self._turn_moves
        if r1 == r2:
            return
        our_role = battle.player_role
        opp_role = "p2" if our_role == "p1" else "p1"
        opp_first = r1 == opp_role
        p1, c1 = _move_info(m1)
        p2, c2 = _move_info(m2)
        if p1 != p2 or c1 not in _DAMAGING or c2 not in _DAMAGING:
            return
        if self._spe_boost[opp_role] != 0:
            return
        opp_species = self._active.get(opp_role)
        our_species = self._active.get(our_role)
        if not opp_species or not our_species:
            return
        our_mon = next((m for m in battle.team.values()
                        if _normalize(m.species) == our_species), None)
        if our_mon is None or not our_mon.stats or not our_mon.stats.get("spe"):
            return
        our_eff = float(our_mon.stats["spe"])
        our_eff *= self._boost_mult(self._spe_boost[our_role])
        if f"{our_role} {our_species}" in self._par:
            our_eff *= 0.5
        if our_role in self._tailwind:
            our_eff *= 2
        # opp side had no boosts (guard above); undo their global modifiers
        # to bound their raw (stat x item) product
        bound = our_eff
        if opp_role in self._tailwind:
            bound /= 2
        if f"{opp_role} {opp_species}" in self._par:
            bound /= 0.5
        if opp_first:
            prev = self.speed_floor.get(opp_species, 0.0)
            self.speed_floor[opp_species] = max(prev, bound)
        else:
            prev = self.speed_ceiling.get(opp_species, float("inf"))
            self.speed_ceiling[opp_species] = min(prev, bound)

    # ---- protocol scan ----

    def update(self, battle):
        replay = getattr(battle, "_replay_data", [])
        our_role = battle.player_role
        opp_role = "p2" if our_role == "p1" else "p1"
        self._opp_role = opp_role
        # our real abilities, for the mold-breaker gate on on-hit evidence
        for m in getattr(battle, "team", {}).values():
            if getattr(m, "ability", None):
                self._our_abilities[_normalize(m.species)] = \
                    _normalize(m.ability)
        for event in replay[self._cursor:]:
            if len(event) < 2:
                continue
            kind = event[1]
            if len(event) >= 3 and str(event[2])[:2] in ("p1", "p2"):
                self._capture_reveals(event, opp_role)
                if kind in ("-damage", "-heal", "-item", "-enditem",
                            "-status", "-activate"):
                    self._note_item_event(event, opp_role)
            if (kind == "-ability" and len(event) >= 4
                    and _normalize(str(event[3])) == "neutralizinggas"):
                # every announce is suppressed under it; the negative-
                # evidence rule stays off for the rest of the battle
                self._nogas = True
                self._ability_latch = None
            if self._ability_latch is not None:
                self._note_ability_announce(event, opp_role)
            # these arrive between |move| and its |-damage|; keep the window open
            if kind not in ("-damage", "-crit", "-supereffective", "-resisted"):
                self._close_pending()
            if kind == "turn":
                self._resolve_hit_latch()   # before upkeep wipes _item_seen
                self._close_turn_upkeep()   # proofs from what did NOT happen
                self._resolve_entry()  # switch-in window closes at end of turn
                self._eval_turn_order(battle)
                self._turn_moves = []
                self._our_attack = None
            elif kind in ("switch", "drag") and len(event) >= 4:
                self._resolve_hit_latch()
                self._resolve_entry()  # a new switch closes the prior window
                role = event[2][:2]
                species = _normalize(event[3].split(",")[0])
                self._active[role] = species
                self._stint_moves.pop(role, None)
                self.sub_hits[role] = []      # a sub does not survive switching
                self._rampage[role] = None    # nor a rampage, nor a charge
                self._charging[role] = None
                self._our_attack = None       # stale across any switch
                if role == opp_role:
                    self._balloon_pending.add(species)
                    # Regenerator proof: back with more HP than it left with
                    if len(event) >= 5 and "/" in str(event[4]):
                        try:
                            cur, mx = (str(event[4]).split(" ")[0]
                                       .split("/")[:2])
                            cur, mx = int(cur), int(mx)
                        except ValueError:
                            cur = mx = 0
                        if cur > 0:
                            stored = self._opp_hp.get(species)
                            if (stored is not None and 0 < stored[0] < cur
                                    and "regenerator"
                                    in _abilities_of(species)):
                                self.regen.add(species)
                            self._opp_hp[species] = (cur, mx)
                    # negative ability evidence: watch the species' FIRST
                    # entry for the announces its dex abilities owe us
                    if species not in self._entered:
                        self._entered.add(species)
                        if not self._nogas:
                            self._ability_latch = {
                                "species": species, "announced": set(),
                                "weather": self._weather,
                                "terrain": self._terrain}
                self._spe_boost[role] = 0
                self._atk_boost[role] = 0
                self._spa_boost[role] = 0
                if role == our_role and "/" in event[4]:
                    try:
                        self._hp[species] = int(event[4].split("/")[0])
                    except ValueError:
                        pass
                # opponent walking into our hazards: open a zero-chip watch
                # (Boots negative evidence). SR is expected on any mon;
                # Spikes only on a species known grounded by type+ability
                # (Air Balloon, the last airborne source, is caught below).
                if role == opp_role:
                    expects_sr = self._side_sr.get(opp_role, False)
                    # Gravity grounds EVERYONE (Flying/Levitate/Air Balloon
                    # all take Spikes while it's up), so it widens the Spikes
                    # check to any species and overrides the balloon exclusion
                    grounded = self._gravity or _grounded_by_species(species)
                    expects_spikes = (self._side_spikes.get(opp_role, False)
                                      and grounded)
                    if expects_sr or expects_spikes:
                        self._entry_latch = {
                            "species": species, "chipped": False,
                            "expects_sr": expects_sr,
                            "expects_spikes": expects_spikes,
                            "gravity": self._gravity, "airborne": False}
            elif kind == "move" and len(event) >= 4:
                self._resolve_hit_latch()
                self._resolve_entry()  # first action after a switch closes it
                role = event[2][:2]
                self._turn_moves.append((role, event[3]))
                mid = _normalize(event[3])
                sp_mv = self._active.get(role) or ""
                self._last_move[role] = (sp_mv, mid)
                # rampage counting: consecutive uses of the SAME lock move
                # by the SAME mon; anything else ends it
                ramp = self._rampage.get(role)
                if mid in _LOCK_MOVES:
                    if ramp and ramp[0] == sp_mv and ramp[1] == mid:
                        self._rampage[role] = (sp_mv, mid, ramp[2] + 1)
                    else:
                        self._rampage[role] = (sp_mv, mid, 1)
                elif ramp is not None:
                    self._rampage[role] = None
                # any move line closes a charge: the charging turn's own
                # |move| precedes its |-prepare| (cleared then re-set), and
                # the completion turn's |move| is the release
                self._charging[role] = None
                called = any(isinstance(a, str) and a.startswith("[from]")
                             for a in event[4:])
                if role == opp_role:
                    _, cat = _move_info(event[3])
                    if cat in _DAMAGING:
                        sp_a = self._active.get(role)
                        if sp_a:
                            self._opp_attacked.add(sp_a)
                    # their own move may self-inflict bare -damage next
                    # (Substitute cost, Belly Drum) — our pending attack
                    # must not claim it
                    self._our_attack = None
                if role == our_role:
                    entry = _moves_data().get(mid, {})
                    # arm the on-hit ability watch: if this move CONNECTS
                    # for direct damage, the reactive abilities stayed
                    # silent and are ruled out at the -damage event
                    self._our_attack = mid \
                        if entry.get("category") in _DAMAGING else None
                    if (entry.get("category") in _DAMAGING
                            and (entry.get("flags") or {}).get("contact")):
                        tgt = self._active.get(opp_role)
                        if tgt:
                            self._we_contacted.add(tgt)
                if role == opp_role and mid != "struggle" and not called:
                    stint = self._stint_moves.setdefault(role, set())
                    stint.add(mid)
                    # Assault Vest forbids SELECTING status moves, so a
                    # genuinely selected one (not [from]-called) disproves
                    # the vest outright (fp cross-audit, 2026-08-04)
                    if _move_info(event[3])[1] not in _DAMAGING:
                        sp_av = self._active.get(role)
                        if sp_av:
                            self._forbid(sp_av, "assaultvest")
                    # a mon that ATTACKS with both categories is not holding a
                    # Choice item: each boosts one category, and locks you into
                    # whichever you clicked, so half the moveset is dead weight
                    # behind a lock. Kyurem's Loaded Dice build is the case that
                    # prompted this — Scale Shot and Icicle Spear physically,
                    # Ice Beam / Freeze-Dry / Earth Power specially. Its Dragon
                    # Dance already rules Choice out via the setup-move gate,
                    # but only once DANCED; the second attacking category shows
                    # up far earlier, and it was during that window that we
                    # branded it Choice Band.
                    if role == opp_role:
                        sp_now = self._active.get(role)
                        if sp_now:
                            seen = self._seen_moves.setdefault(sp_now, set())
                            seen.add(mid)
                            if len(_offensive_categories(seen)) >= 2:
                                # Band and Specs ONLY. Choice Scarf boosts
                                # SPEED, which every move uses equally, so a
                                # mixed attacker wastes none of it — and the
                                # canonical mixed Scarf set is Iron Valiant
                                # (Close Combat + Moonblast), the very mon in
                                # this module's header whose real Scarf we
                                # kept mis-modelling as Booster Energy.
                                # Forbidding it here would re-break the case
                                # set_inference was written for.
                                self._forbid(sp_now, "choiceband",
                                             "choicespecs")
                                if self.confirmed.get(sp_now) in (
                                        "choiceband", "choicespecs"):
                                    del self.confirmed[sp_now]
                    if len(stint) >= 2:
                        sp = self._active.get(role)
                        if sp:
                            self.choice_disproven.add(sp)
                            if self.confirmed.get(sp) in (
                                    "choiceband", "choicespecs",
                                    "choicescarf"):
                                del self.confirmed[sp]
                if role == opp_role:
                    target = self._active.get(our_role)
                    clean = (self._atk_boost[opp_role] <= 0
                             and self._spa_boost[opp_role] <= 0
                             and opp_role not in self._tera)
                    self._pending = {
                        "attacker": self._active.get(opp_role),
                        "move": _normalize(event[3]), "target": target,
                        "damage": 0, "invalid": not clean,
                        "weather": self._weather, "se": False,
                    }
            elif kind == "-damage" and len(event) >= 4:
                role = event[2][:2]
                # on-hit negative evidence: our armed attack landed DIRECT
                # damage ([from]-tagged lines are hazards/status/item
                # residuals, not the hit). Abilities convict immediately —
                # their announce would have REPLACED this damage line.
                # Items latch instead: their announces (eat/boost/enditem)
                # FOLLOW the damage, so candidates arm here and convict at
                # the next commitment point via _resolve_hit_latch.
                if (role == opp_role and self._our_attack
                        and not self._nogas
                        and not any(isinstance(a, str)
                                    and a.startswith("[from]")
                                    for a in event[4:])):
                    e = _moves_data().get(self._our_attack, {})
                    mtype = str(e.get("type", "")).lower()
                    sp_hit = self._active.get(opp_role)
                    pierced = self._our_abilities.get(
                        self._active.get(our_role) or "") in _ABILITY_IGNORERS
                    if sp_hit and not pierced:
                        bad = list(_TYPE_REACTIVE.get(mtype, ()))
                        fl = e.get("flags") or {}
                        for f, abs_ in _FLAG_REACTIVE.items():
                            if fl.get(f):
                                bad += abs_
                        if (e.get("priority", 0) or 0) > 0:
                            bad += _PRIORITY_REACTIVE
                        if bad:
                            self.impossible_abilities.setdefault(
                                sp_hit, set()).update(bad)
                    prev = self._opp_hp.get(sp_hit) if sp_hit else None
                    head = str(event[3]).split(" ")[0]
                    post = None
                    if head.startswith("0"):
                        post = (0, prev[1] if prev else 100)
                    elif "/" in head:
                        try:
                            cur, mx = head.split("/")
                            post = (int(cur), int(mx))
                        except ValueError:
                            post = None
                    if sp_hit and post is not None:
                        if post[0] <= 0:
                            # KO from FULL HP is itself the Focus Sash
                            # proof (the sash would have held at 1) — and
                            # the Sturdy proof, ability-pierce gated.
                            # Multihit breaks the sash and KOs anyway.
                            if (prev is not None and prev[0] >= prev[1]
                                    and not e.get("multihit")):
                                self._forbid(sp_hit, "focussash")
                                if not pierced:
                                    self.impossible_abilities.setdefault(
                                        sp_hit, set()).add("sturdy")
                        else:
                            cands = {"ejectbutton", "redcard"}
                            if self._our_se:
                                cands.add("weaknesspolicy")
                            cands.update(_ONHIT_TYPE_ITEMS.get(mtype, ()))
                            # Unnerve (ours) suppresses ALL their berries —
                            # an uneaten berry under it proves nothing
                            if self._our_abilities.get(
                                    self._active.get(our_role) or "") \
                                    != "unnerve":
                                if self._our_se:
                                    wb = _WEAKNESS_BERRIES.get(mtype)
                                    if wb:
                                        cands.add(wb)
                                if mtype == "normal":
                                    cands.add("chilanberry")
                                frac = post[0] / post[1] if post[1] else 1.0
                                if frac <= 0.5:
                                    cands.update(_PINCH_50)
                                if frac <= 0.25:
                                    cands.update(_PINCH_25)
                            self._hit_latch = {"species": sp_hit,
                                               "items": cands}
                    self._our_attack = None
                    self._our_se = False
                # opponent's incoming mon took hazard chip -> it does NOT
                # have Boots; cancel the negative-evidence latch
                if role == opp_role and any(
                        ("Stealth Rock" in a or "Spikes" in a)
                        for a in event[4:]):
                    if self._entry_latch is not None:
                        self._entry_latch["chipped"] = True
                    # hazard chip PROVES no Boots. Measured rare (0.2% of
                    # assumptions) because the mons that take chip are the
                    # ones whose builds do not run Boots anyway — kept
                    # because it costs one line on the constraint layer.
                    self._forbid(self._event_species(event) or "",
                                 "heavydutyboots")
                if role == opp_role and "/" in str(event[3]):
                    sp = self._event_species(event)
                    try:
                        cur, mx = event[3].split(" ")[0].split("/")
                        if sp:
                            self._opp_hp[sp] = (int(cur), int(mx))
                    except ValueError:
                        pass
                    if sp and len(event[3].split(" ")) == 1:
                        self._opp_unstatused.add(sp)
                    elif sp:
                        self._opp_unstatused.discard(sp)
                if role == our_role:
                    species = self._event_species(event) or ""
                    new_hp = 0
                    if "/" in event[3]:
                        try:
                            new_hp = int(event[3].split("/")[0])
                        except ValueError:
                            new_hp = 0
                    prev = self._hp.get(species)
                    if (self._pending is not None
                            and self._pending["target"] == species
                            and prev is not None):
                        self._pending["damage"] += max(0, prev - new_hp)
                    self._hp[species] = new_hp
            elif kind == "-heal" and len(event) >= 4:
                role = event[2][:2]
                if role == opp_role:
                    sp = self._event_species(event)
                    if sp:
                        self._healed_this_turn.add(sp)
                        if "/" in str(event[3]):
                            try:
                                cur, mx = event[3].split(" ")[0].split("/")
                                self._opp_hp[sp] = (int(cur), int(mx))
                            except ValueError:
                                pass
                if role == our_role and "/" in event[3]:
                    species = self._event_species(event)
                    if species:
                        try:
                            self._hp[species] = int(event[3].split("/")[0])
                        except ValueError:
                            pass
            elif kind == "faint" and len(event) >= 3:
                # zero the stored HP so a later Revival Blessing return
                # (alive again, higher hp) can't fake a Regenerator proof
                if event[2][:2] == opp_role:
                    sp = self._event_species(event)
                    if sp:
                        self._opp_hp[sp] = (0, self._opp_hp.get(
                            sp, (0, 100))[1])
            elif kind == "-sethp" and len(event) >= 4:
                # Pain Split writes HP with no -damage/-heal; without this
                # the stored value goes stale and the regen compare lies
                if event[2][:2] == opp_role and "/" in str(event[3]):
                    sp = self._event_species(event)
                    try:
                        cur, mx = str(event[3]).split(" ")[0].split("/")
                        if sp:
                            self._opp_hp[sp] = (int(cur), int(mx))
                    except ValueError:
                        pass
            elif kind == "-prepare" and len(event) >= 4:
                # a charge turn: committed (and often semi-invulnerable)
                # until the release |move| line, a switch, a full stop
                # (cant), or a Power Herb instant release. A confusion
                # self-hit on the release turn emits none of those, so the
                # state can run one turn stale there — it self-corrects on
                # the mon's next move line.
                self._charging[event[2][:2]] = _normalize(str(event[3]))
            elif kind == "cant" and len(event) >= 3:
                role = event[2][:2]
                self._rampage[role] = None    # a full stop ends a rampage
                self._charging[role] = None   # and interrupts a charge
            elif kind == "-miss" and len(event) >= 3:
                # gen5+: a missed rampage turn ends the rampage
                self._rampage[event[2][:2]] = None
            elif kind == "-enditem" and len(event) >= 4:
                if "Power Herb" in str(event[3]):
                    self._charging[event[2][:2]] = None
            elif kind == "-start" and len(event) >= 4:
                role = event[2][:2]
                if "Substitute" in str(event[3]):
                    self.sub_hits[role] = []          # a fresh sub
                elif ("confusion" in str(event[3]).lower()
                        and any(a == "[fatigue]" for a in event[4:]
                                if isinstance(a, str))):
                    self._rampage[role] = None        # rampage fatigued out
            elif kind == "-end" and len(event) >= 4:
                if "Substitute" in str(event[3]):
                    self.sub_hits[event[2][:2]] = []  # sub broke
            elif kind == "-activate" and len(event) >= 4:
                # a sub absorbed a hit; the amount is deliberately hidden by
                # the protocol, so record WHICH move hit it (the other
                # role's last move line) for downstream damage estimation
                if ("Substitute" in str(event[3])
                        and any(a == "[damage]" for a in event[4:]
                                if isinstance(a, str))):
                    role = event[2][:2]
                    other = "p2" if role == "p1" else "p1"
                    lm = self._last_move.get(other)
                    if lm and lm[0]:
                        self.sub_hits[role].append(lm)
            elif kind == "-status" and len(event) >= 3:
                role = event[2][:2]
                sp = self._event_species(event)
                if role == opp_role:
                    self._opp_unstatused.discard(sp or "")
                # par tracking used to live in a LATER elif of this same
                # chain — dead code, every -status matched here first, so
                # _par stayed empty and the speed-order par corrections
                # never fired (found 2026-08-04)
                if sp and len(event) >= 4 and event[3] == "par":
                    self._par.add(f"{role} {sp}")
            elif kind == "-curestatus" and len(event) >= 3:
                role = event[2][:2]
                sp = self._event_species(event)
                if role == opp_role and sp:
                    self._opp_unstatused.add(sp)
                if sp and len(event) >= 4 and event[3] == "par":
                    self._par.discard(f"{role} {sp}")
            elif kind == "-crit":
                if self._pending is not None:
                    self._pending["invalid"] = True
            elif kind == "-supereffective":
                if self._pending is not None:
                    self._pending["se"] = True
                if (len(event) >= 3 and str(event[2])[:2] == opp_role
                        and self._our_attack):
                    self._our_se = True    # precedes the -damage line
            elif kind in ("-boost", "-unboost") and len(event) >= 5:
                role = event[2][:2]
                try:
                    delta = int(event[4]) * (1 if kind == "-boost" else -1)
                except ValueError:
                    delta = 0
                stat = event[3]
                if stat == "spe":
                    self._spe_boost[role] += delta
                elif stat == "atk":
                    self._atk_boost[role] += delta
                elif stat == "spa":
                    self._spa_boost[role] += delta
            elif kind == "-terastallize":
                self._tera.add(event[2][:2])
            elif kind == "-sidestart" and len(event) >= 4:
                if "Tailwind" in event[3]:
                    self._tailwind.add(event[2][:2])
                elif "Stealth Rock" in event[3]:
                    self._side_sr[event[2][:2]] = True
                elif "Spikes" in event[3] and "Toxic Spikes" not in event[3]:
                    self._side_spikes[event[2][:2]] = True
            elif kind == "-sideend" and len(event) >= 4:
                if "Tailwind" in event[3]:
                    self._tailwind.discard(event[2][:2])
                elif "Stealth Rock" in event[3]:
                    self._side_sr[event[2][:2]] = False
                elif "Spikes" in event[3] and "Toxic Spikes" not in event[3]:
                    self._side_spikes[event[2][:2]] = False
            elif kind == "-swapsideconditions":
                # Court Change flips hazards to the opposite sides
                self._side_sr = {"p1": self._side_sr["p2"],
                                 "p2": self._side_sr["p1"]}
                self._side_spikes = {"p1": self._side_spikes["p2"],
                                     "p2": self._side_spikes["p1"]}
            elif (kind == "-item" and len(event) >= 4
                  and self._entry_latch is not None
                  and event[2][:2] == opp_role
                  and "Air Balloon" in event[3]):
                # Air Balloon announces on switch-in; it makes the holder
                # airborne for Spikes (but NOT for Stealth Rock), so it only
                # voids Spikes-based evidence
                self._entry_latch["airborne"] = True
            elif kind == "-fieldstart" and len(event) >= 3:
                if "Trick Room" in event[2]:
                    self._trick_room = True
                elif "Gravity" in event[2]:
                    self._gravity = True
                else:
                    for tname, tid in (("Electric Terrain", "electric"),
                                       ("Grassy Terrain", "grassy"),
                                       ("Misty Terrain", "misty"),
                                       ("Psychic Terrain", "psychic")):
                        if tname in str(event[2]):
                            self._terrain = tid
            elif kind == "-fieldend" and len(event) >= 3:
                if "Trick Room" in event[2]:
                    self._trick_room = False
                elif "Gravity" in event[2]:
                    self._gravity = False
                elif "Terrain" in str(event[2]):
                    self._terrain = "none"
            elif kind == "-weather" and len(event) >= 3:
                self._weather = {"SunnyDay": "sun", "RainDance": "rain",
                                 "Sandstorm": "sand", "Snowscape": "snow",
                                 "Snow": "snow", "Hail": "hail",
                                 "none": "none"}.get(event[2], "none")
        self._resolve_entry()  # the final switch-in has no following event
        self._cursor = len(replay)

    # ---- decisions ----

    def boots_inferred(self, species: str) -> str | None:
        """'heavydutyboots' when a switch-in over our Stealth Rock took no
        chip AND the species cannot run Magic Guard. The ambiguous (Magic-
        Guard-capable) case is deliberately NOT promoted — the search must
        not model the wrong item — though it is recorded in boots_ambiguous
        for a hedged commentary read."""
        return "heavydutyboots" if species in self.boots else None

    def scarf_needed(self, species: str, modeled_spe: int, item: str) -> bool:
        """True if the modeled (stat, item) contradicts an observed floor."""
        floor = self.speed_floor.get(species)
        if floor is None:
            return False
        mult = 1.5 if item in ("choicescarf",) else 1.0
        return modeled_spe * mult < floor

    def max_speed_needed(self, species: str, scarfed_spe: int) -> bool:
        floor = self.speed_floor.get(species)
        return floor is not None and scarfed_spe * 1.5 < floor

    def max_speed_suffices(self, species: str, max_spe: int) -> bool:
        """True when FULL Speed investment alone clears the observed floor.

        The cheaper hypothesis, and it must be tested BEFORE inferring a Choice
        Scarf: "they ran more Speed than my canonical spread assumed" is far
        more common than "they are holding a Scarf", especially for species
        whose standard sets are bulky with little or no Speed. Caught live
        2026-07-31 — a Slowking-Galar outsped our PARALYSED Iron Crown (~162
        effective), which a max-invested no-item Slowking (~174) clears on its
        own, and the engine announced a Scarf anyway.

        Getting this order wrong is worse than a cosmetic mislabel: believing
        Choice Scarf also means believing the target is CHOICE-LOCKED, so the
        search plans against one move it may switch off freely.
        """
        floor = self.speed_floor.get(species)
        return floor is not None and max_spe >= floor

    def speed_clamp(self, species: str, modeled_spe: int, item: str) -> tuple[int, str] | None:
        """(clamped_stat, item) when a ceiling contradicts the model: drop an
        inferred scarf first, then clamp the raw stat (covers slower spreads
        and speed-drop items like Iron Ball without naming them). Not applied
        when a floor exists — floors are the sweep-killers, ceilings advisory."""
        ceil = self.speed_ceiling.get(species)
        if ceil is None or species in self.speed_floor:
            return None
        mult = 1.5 if item == "choicescarf" else 1.0
        if modeled_spe * mult < ceil:
            return None
        if item == "choicescarf":
            item = "none"
        if modeled_spe >= ceil:
            return max(1, int(ceil) - 1), item
        return modeled_spe, item

    def _observed_ratio(self, ev: dict, opp_mon: pe.Pokemon,
                        our_mons: dict) -> float | None:
        """observed damage / modeled max roll for one observation.

        `opp_mon` should be the MAX-INVESTED probe, not the canonical-spread
        one — see damage_item_upgrade for why."""
        our = our_mons.get(ev["our_species"])
        if our is None:
            return None
        try:
            state = pe.State(
                side_one=pe.Side(pokemon=[opp_mon] + [
                    pe.Pokemon.create_fainted() for _ in range(5)]),
                side_two=pe.Side(pokemon=[our] + [
                    pe.Pokemon.create_fainted() for _ in range(5)]),
                weather={"sun": pe.Weather.SUN, "rain": pe.Weather.RAIN,
                         "sand": pe.Weather.SAND, "snow": pe.Weather.SNOW,
                         "hail": pe.Weather.HAIL}.get(ev["weather"],
                                                      pe.Weather.NONE),
                weather_turns_remaining=3 if ev["weather"] != "none" else 0,
            )
            # returns (side_one_move_rolls, side_two_move_rolls)
            rolls = pe.calculate_damage(state, ev["move"], "splash", True)[0]
        except Exception:
            return None
        if not rolls or max(rolls) <= 0:
            return None
        return ev["damage"] / max(rolls)

    def damage_item_upgrade(self, species: str, opp_mon: pe.Pokemon,
                            our_mons: dict[str, pe.Pokemon],
                            known_moves: tuple[str, ...] = (),
                            invested_probe: pe.Pokemon | None = None) -> str | None:
        """Weakest damage-boosting item consistent with ALL observations.

        Ratio brackets over the max roll: >1.38 -> Choice Band/Specs (1.5x);
        1.15-1.38 -> Life Orb (1.3x), unless the boost is confined to one
        move type while another damaging type reads clean (<=1.05), in which
        case a 1.2x type item is inferred. Expert Belt and Booster Energy
        land in the Life Orb bracket and are modeled as it — the damage
        multiplier is what the search needs, not the item's name.

        CHEAPEST EXPLANATION FIRST (2026-08-03). Those thresholds are derived
        from ITEM multipliers, so they only mean what they claim when the
        denominator is THE SAME MON HOLDING NOTHING. It used to be the max
        roll of our CANONICAL-SPREAD guess, which conflates two different
        errors: "they hold an item" and "they invested where my guess did
        not". The second is far more common — any mon meant to attack is
        invested in Atk or SpA — so the ratio was measuring our spread error
        and billing it to an item.

        How wrong: across the 392 (set, attacking-stat) pairs in the gen9ou
        curated corpus, investment ALONE clears the Life Orb bracket for
        68.9% and the CHOICE bracket for 57.9%. Restricted to stats a set
        actually attacks with it is 46.9% / 28.1%; for a category the
        canonical set does not invest in at all — a coverage move, or simply
        an offensive variant of a mon we model defensively — it is 99.4%,
        i.e. a guaranteed false Choice brand. And when there is no canonical
        set at all the fallback spread (31 IV / 85 EV / neutral) sits a
        median 1.32x below max investment, so a plain Life Orb attacker
        reads 1.71x and gets branded Choice Band.

        The fix is the one the Speed path already uses (max_speed_suffices):
        exhaust the free explanation before claiming an item. `invested_probe`
        is the same Pokemon with Atk/SpA/Def at 252 EV / 31 IV / +nature, and
        only damage exceeding THAT needs an item to explain it. The brackets
        are unchanged and are now correct rather than merely calibrated: a
        genuine Band on a max-invested attacker still reads 1.5x, a genuine
        Orb still 1.3x. Maxing all three stats at once is EV-illegal, but
        only one is read per damage calc, so each read is a legal maximum;
        Def is included so Body Press cannot sneak through the same hole.
        Kill switch: CB_DAMAGE_INVESTED=0 restores the canonical denominator.

        `known_moves` gates the Choice bracket: a mon with a revealed status
        move is never branded Choice-locked, however hot the hit — the
        audited Gliscor game had a beyond-roll Knock Off (an invested
        spread, most likely) upgrade a Protect/Toxic mon to Choice Band in
        every world for the rest of the game. That gate was the band-aid over
        this same bug and stays: it catches the sets whose revealed moves
        prove no lock, while the invested denominator catches the rest.
        """
        if invested_probe is not None and _INVESTED_DENOM:
            opp_mon = invested_probe
        boosted: list[dict] = []
        clean: list[dict] = []
        for ev in self.damage_evidence:
            if ev["species"] != species:
                continue
            _, category = _move_info(ev["move"])
            if category not in _DAMAGING:
                continue
            ratio = self._observed_ratio(ev, opp_mon, our_mons)
            if ratio is None:
                continue
            entry = {"type": _move_type(ev["move"]), "category": category,
                     "ratio": ratio, "se": ev.get("se", False)}
            if ratio > 1.15:
                boosted.append(entry)
            elif ratio <= 1.05:
                clean.append(entry)
        if not boosted:
            return None
        from showdown.chaos_stats import incompatible_items
        choice_ok = ("choiceband" not in incompatible_items(known_moves)
                     and species not in self.choice_disproven)
        top = max(boosted, key=lambda e: e["ratio"])
        if top["ratio"] > 1.38 and choice_ok:
            return "choiceband" if top["category"] == "Physical" else "choicespecs"
        # 1.2x bracket disambiguation: boost confined to SE hits -> Expert
        # Belt; confined to one type with another type clean -> type item
        if (all(e["se"] for e in boosted)
                and any(not e["se"] for e in clean)):
            return "expertbelt"
        boosted_types = {e["type"] for e in boosted}
        clean_types = {e["type"] for e in clean}
        if len(boosted_types) == 1 and (clean_types - boosted_types):
            return _TYPE_ITEM.get(next(iter(boosted_types)), "lifeorb")
        return "lifeorb"
