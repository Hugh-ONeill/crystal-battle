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
#   - DAMAGE BRACKETS: a non-crit, boost-free, non-tera hit that exceeds the
#     modeled set's MAXIMUM roll by >15% proves a boosting item. The WEAKEST
#     item that explains the hit is chosen: Life Orb (1.3x, boosts both
#     categories) up to ~1.38x over max roll, Choice Band/Specs beyond.
#
# Observations only ever apply to inferred details — revealed items are
# never overridden.

from __future__ import annotations

import poke_engine as pe

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

    def _close_pending(self):
        p = self._pending
        self._pending = None
        if p is None or p["damage"] <= 0 or p["invalid"]:
            return
        self.damage_evidence.append({
            "species": p["attacker"], "move": p["move"],
            "damage": p["damage"], "our_species": p["target"],
            "weather": p["weather"], "se": p["se"],
        })

    def _resolve_entry(self):
        """A switch-in over hazards closes its window (the next turn/move/
        switch after the immediate entry damage). Zero chip is Boots — or
        Magic Guard if the species can run it. Evidence is airtight for
        Stealth Rock (hits everything); for Spikes it only counts if the mon
        wasn't revealed airborne by an Air Balloon on entry (the species is
        already known grounded — that's the gate to latch on Spikes)."""
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
        for event in replay[self._cursor:]:
            if len(event) < 2:
                continue
            kind = event[1]
            # these arrive between |move| and its |-damage|; keep the window open
            if kind not in ("-damage", "-crit", "-supereffective", "-resisted"):
                self._close_pending()
            if kind == "turn":
                self._resolve_entry()  # switch-in window closes at end of turn
                self._eval_turn_order(battle)
                self._turn_moves = []
            elif kind in ("switch", "drag") and len(event) >= 4:
                self._resolve_entry()  # a new switch closes the prior window
                role = event[2][:2]
                species = _normalize(event[3].split(",")[0])
                self._active[role] = species
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
                self._resolve_entry()  # first action after a switch closes it
                role = event[2][:2]
                self._turn_moves.append((role, event[3]))
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
                # opponent's incoming mon took hazard chip -> it does NOT
                # have Boots; cancel the negative-evidence latch
                if (role == opp_role and self._entry_latch is not None
                        and any(("Stealth Rock" in a or "Spikes" in a)
                                for a in event[4:])):
                    self._entry_latch["chipped"] = True
                if role == our_role:
                    species = _normalize(event[2].split(":")[1])
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
                if role == our_role and "/" in event[3]:
                    species = _normalize(event[2].split(":")[1])
                    try:
                        self._hp[species] = int(event[3].split("/")[0])
                    except ValueError:
                        pass
            elif kind == "-crit":
                if self._pending is not None:
                    self._pending["invalid"] = True
            elif kind == "-supereffective":
                if self._pending is not None:
                    self._pending["se"] = True
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
            elif kind == "-status" and len(event) >= 4 and event[3] == "par":
                role = event[2][:2]
                species = _normalize(event[2].split(":")[1])
                self._par.add(f"{role} {species}")
            elif kind == "-curestatus" and len(event) >= 4 and event[3] == "par":
                role = event[2][:2]
                species = _normalize(event[2].split(":")[1])
                self._par.discard(f"{role} {species}")
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
            elif kind == "-fieldend" and len(event) >= 3:
                if "Trick Room" in event[2]:
                    self._trick_room = False
                elif "Gravity" in event[2]:
                    self._gravity = False
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
        """observed damage / modeled max roll for one observation."""
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
                            our_mons: dict[str, pe.Pokemon]) -> str | None:
        """Weakest damage-boosting item consistent with ALL observations.

        Ratio brackets over the modeled max roll: >1.38 -> Choice Band/Specs
        (1.5x); 1.15-1.38 -> Life Orb (1.3x), unless the boost is confined to
        one move type while another damaging type reads clean (<=1.05), in
        which case a 1.2x type item is inferred. Expert Belt and Booster
        Energy land in the Life Orb bracket and are modeled as it — the
        damage multiplier is what the search needs, not the item's name.
        """
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
        top = max(boosted, key=lambda e: e["ratio"])
        if top["ratio"] > 1.38:
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
