# Pokemon Crystal battle AI reimplementation
# Based on pret/pokecrystal engine/battle/ai/ (move.asm, scoring.asm, switch.asm)
# Usage: CrystalAIAgent(layers=AI_CHAMPION) for gym leader-tier AI

from __future__ import annotations

import random

from engine.actions import Action, Struggle, Switch, UseMove
from engine.damage import calc_expected_damage
from engine.player_state import PlayerState
from engine.pokemon import Pokemon
from engine.stat_stages import MOVE_STAT_EFFECTS
from engine.status import can_apply_status, effective_speed
from engine.types import TypeChart


# ============================================================
# LAYER BIT FLAGS
# ============================================================

AI_BASIC       = 0x001  # filter redundant/impossible moves
AI_SETUP       = 0x002  # encourage boosts early, discourage late
AI_TYPES       = 0x004  # type effectiveness scoring
AI_OFFENSIVE   = 0x008  # discourage non-damaging moves
AI_SMART       = 0x010  # 65+ effect-specific heuristics
AI_OPPORTUNIST = 0x020  # discourage status when low HP
AI_AGGRESSIVE  = 0x040  # real damage calc, pick highest
AI_CAUTIOUS    = 0x080  # discourage status after turn 1
AI_STATUS      = 0x100  # status immunity awareness
AI_RISKY       = 0x200  # KO detection bonus

# ---- Presets ----
# gym leaders, E4, champion, rival, Red (7 layers)
AI_CHAMPION = AI_BASIC | AI_SETUP | AI_SMART | AI_AGGRESSIVE | AI_CAUTIOUS | AI_STATUS | AI_RISKY
# executives, scientists, sages (5 layers)
AI_EXECUTIVE = AI_BASIC | AI_TYPES | AI_SMART | AI_OFFENSIVE | AI_OPPORTUNIST
# youngsters, bug catchers (3 layers)
AI_TRAINER = AI_BASIC | AI_CAUTIOUS | AI_STATUS
# all layers active
AI_ALL = 0x3FF


# ============================================================
# MOVE CATEGORIES
# ============================================================

_RECOVERY_MOVES = frozenset({
    "Recover", "Softboiled", "Morning Sun", "Synthesis", "Milk Drink", "Moonlight",
})
_SELF_KO_MOVES = frozenset({"Explosion", "Self-Destruct", "Selfdestruct"})
_WEATHER_MOVES = {"Rain Dance": "rain", "Sunny Day": "sun", "Sandstorm": "sandstorm"}
_SLEEP_COMBO_MOVES = frozenset({"Dream Eater", "Nightmare"})
_PROTECT_MOVES = frozenset({"Protect", "Detect"})
_PHAZE_MOVES = frozenset({"Whirlwind", "Roar"})

# ailment_id -> engine status string
_AILMENT_MAP = {1: "par", 2: "slp", 3: "frz", 4: "brn", 5: "psn", 6: "tox"}


# ============================================================
# CRYSTAL AI AGENT
# ============================================================

class CrystalAIAgent:
    """Pokemon Crystal battle AI with configurable scoring layers.

    Move scoring: each move starts at 20 (lower = better). Layers add/subtract
    based on heuristics. AI picks the minimum score (random tiebreak).
    Switching evaluated before move selection.
    """

    def __init__(
        self,
        type_chart: TypeChart | None = None,
        layers: int = AI_CHAMPION,
        seed: int | None = None,
        switch_chance: float = 0.5,
    ):
        self._tc = type_chart or TypeChart.load()
        self._layers = layers
        self._rng = random.Random(seed)
        self._switch_chance = switch_chance
        self._turn = 0
        self._prev_active_name: str | None = None

    def _chance(self, pct: float) -> bool:
        """Roll a probability gate (matches Crystal's AI_50_50, AI_80_20, etc.)."""
        return self._rng.random() < pct

    def act(self, my_state: PlayerState, opp_state: PlayerState) -> Action:
        me = my_state.active
        # track per-matchup turn count (resets on switch)
        if me.name != self._prev_active_name:
            self._turn = 0
            self._prev_active_name = me.name
        self._turn += 1

        # forced switch
        if my_state.must_switch:
            return self._pick_switch(my_state, opp_state)

        if not me.has_any_pp():
            return Struggle()

        # evaluate switching before move scoring
        if self._should_switch(my_state, opp_state):
            switch = self._pick_switch(my_state, opp_state)
            if switch is not None:
                return switch

        # ---- Score all 4 moves ----
        scores = self._score_moves(my_state, opp_state)

        # pick minimum score among valid moves (random tiebreak)
        best = float("inf")
        candidates: list[int] = []
        for i, slot in enumerate(me.move_slots):
            if not slot.has_pp:
                continue
            if scores[i] < best:
                best = scores[i]
                candidates = [i]
            elif scores[i] == best:
                candidates.append(i)

        if not candidates:
            return Struggle()
        return UseMove(slot_index=self._rng.choice(candidates))

    # ============================================================
    # SCORING PIPELINE
    # ============================================================

    def _score_moves(self, my: PlayerState, opp: PlayerState) -> list[float]:
        me = my.active
        them = opp.active
        scores = [20.0, 20.0, 20.0, 20.0]

        # pad if fewer than 4 moves
        n_moves = len(me.move_slots)
        for i in range(4):
            if i >= n_moves or not me.move_slots[i].has_pp:
                scores[i] = 80.0

        layers = self._layers
        if layers & AI_BASIC:
            self._layer_basic(scores, me, them, my, opp)
        if layers & AI_SETUP:
            self._layer_setup(scores, me)
        if layers & AI_TYPES:
            self._layer_types(scores, me, them)
        if layers & AI_OFFENSIVE:
            self._layer_offensive(scores, me)
        if layers & AI_SMART:
            self._layer_smart(scores, me, them, my, opp)
        if layers & AI_OPPORTUNIST:
            self._layer_opportunist(scores, me)
        if layers & AI_AGGRESSIVE:
            self._layer_aggressive(scores, me, them, opp)
        if layers & AI_CAUTIOUS:
            self._layer_cautious(scores, me)
        if layers & AI_STATUS:
            self._layer_status(scores, me, them)
        if layers & AI_RISKY:
            self._layer_risky(scores, me, them, opp)

        return scores

    # ---- AI_Basic: redundant move filtering ----

    def _layer_basic(self, scores, me, them, my, opp):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            mv = me.move_slots[i].template
            name = mv.name

            # recovery at full HP
            if (name in _RECOVERY_MOVES or name == "Rest") and me.hp_frac >= 1.0:
                scores[i] += 10

            # status move on already-statused target
            if mv.damage_class == "status" and mv.meta:
                aid = mv.meta.get("ailment_id", 0)
                if aid > 0 and them.status is not None:
                    scores[i] += 10

            # confuse on already confused
            if name in ("Confuse Ray", "Swagger", "Sweet Kiss") and them.confusion_turns > 0:
                scores[i] += 10

            # leech seed on already seeded
            if name == "Leech Seed" and them.leech_seeded:
                scores[i] += 10

            # screens already active
            if name == "Reflect" and my.side.reflect_turns > 0:
                scores[i] += 10
            if name == "Light Screen" and my.side.light_screen_turns > 0:
                scores[i] += 10

            # spikes already up
            if name == "Spikes" and opp.side.spikes:
                scores[i] += 10

            # dream eater / nightmare on non-sleeping target
            if name in _SLEEP_COMBO_MOVES and them.status != "slp":
                scores[i] += 10

    # ---- AI_Setup: boost timing ----

    def _layer_setup(self, scores, me):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            name = me.move_slots[i].template.name
            if name not in MOVE_STAT_EFFECTS:
                continue
            effects = MOVE_STAT_EFFECTS[name]
            is_boost = any(t == "self" and s > 0 for _, s, t in effects)
            is_debuff = any(t == "opponent" and s < 0 for _, s, t in effects)
            if not (is_boost or is_debuff):
                continue

            if self._turn <= 1:
                # first turn: 50% chance to encourage setup
                if self._chance(0.5):
                    scores[i] -= 2
            else:
                # later turns: 90% chance to discourage
                if self._chance(0.9):
                    scores[i] += 2

    # ---- AI_Types: type effectiveness ----

    def _layer_types(self, scores, me, them):
        has_damaging = any(
            s.has_pp and s.template.power > 0 for s in me.move_slots
        )
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            mv = me.move_slots[i].template
            if mv.power == 0:
                continue
            eff = self._tc.combined_effectiveness(mv.type, them.types)
            if eff >= 2.0:
                scores[i] -= 1
            elif eff == 0.0:
                scores[i] += 10
            elif eff <= 0.5 and has_damaging:
                scores[i] += 1

    # ---- AI_Offensive: pure aggro ----

    def _layer_offensive(self, scores, me):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            if me.move_slots[i].template.power == 0:
                scores[i] += 2

    # ---- AI_Smart: effect-specific heuristics ----

    def _layer_smart(self, scores, me, them, my, opp):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            mv = me.move_slots[i].template
            name = mv.name

            # ---- Recovery ----
            if name in _RECOVERY_MOVES or name == "Rest":
                if me.hp_frac < 0.25 and self._chance(0.9):
                    scores[i] -= 2
                elif me.hp_frac > 0.5:
                    scores[i] += 1

            # ---- Sleep combos ----
            if name in _SLEEP_COMBO_MOVES and them.status == "slp":
                scores[i] -= 3

            # ---- Status moves ----
            if mv.damage_class == "status" and mv.meta:
                aid = mv.meta.get("ailment_id", 0)

                # paralyze: encourage if slower
                if aid == 1 and them.status is None:
                    if effective_speed(me) < effective_speed(them) and self._chance(0.8):
                        scores[i] -= 2
                    if them.hp_frac < 0.25:
                        scores[i] += 1

                # sleep: strongly encourage
                if aid == 2 and them.status is None and self._chance(0.8):
                    scores[i] -= 3

                # poison/toxic: less useful on low-HP targets
                if aid in (5, 6) and them.hp_frac < 0.5:
                    scores[i] += 1

                # confuse: discourage on low-HP targets
                if aid == 7:
                    if them.hp_frac < 0.5 and self._chance(0.9):
                        scores[i] += 1
                    if them.hp_frac < 0.25:
                        scores[i] += 1

            # ---- Stat boosts: check if already maxed ----
            if name in MOVE_STAT_EFFECTS:
                for stat, stages, target in MOVE_STAT_EFFECTS[name]:
                    if target == "self" and stages > 0:
                        current = me.stat_stages.get(stat, 0)
                        if current >= 4:
                            scores[i] += 2
                        elif me.hp_frac < 0.3:
                            scores[i] += 1

            # ---- Belly Drum ----
            if name == "Belly Drum":
                if me.stat_stages.get("attack", 0) >= 2:
                    scores[i] += 5
                if me.hp_frac < 0.5:
                    scores[i] += 5

            # ---- Priority KO ----
            if mv.priority > 0 and mv.power > 0:
                dmg = calc_expected_damage(me, them, mv, self._tc, screens=opp.side)
                if dmg >= them.current_hp:
                    scores[i] -= 3
                elif effective_speed(me) > effective_speed(them):
                    scores[i] += 1

            # ---- Self-destruct ----
            if name in _SELF_KO_MOVES:
                alive = sum(1 for p in my.team if not p.is_fainted)
                if alive <= 1:
                    scores[i] += 3
                if me.hp_frac > 0.25:
                    scores[i] += 3

            # ---- Protect ----
            if name in _PROTECT_MOVES:
                if me.protect_consecutive > 0:
                    scores[i] += 3
                if them.status in ("psn", "tox") or them.leech_seeded:
                    if self._chance(0.8):
                        scores[i] -= 1

            # ---- Phazing (Whirlwind/Roar) ----
            if name in _PHAZE_MOVES:
                if any(v > 0 for v in them.stat_stages.values()):
                    scores[i] -= 2
                else:
                    scores[i] += 1

    # ---- AI_Opportunist: finish them off, don't status ----

    def _layer_opportunist(self, scores, me):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            if me.move_slots[i].template.power == 0:
                if me.hp_frac < 0.25:
                    scores[i] += 1
                elif me.hp_frac < 0.5 and self._chance(0.5):
                    scores[i] += 1

    # ---- AI_Aggressive: real damage calc, best move wins ----

    def _layer_aggressive(self, scores, me, them, opp):
        best_dmg = -1.0
        best_idx = -1
        dmgs: list[float] = []
        for i in range(len(me.move_slots)):
            if scores[i] >= 80 or me.move_slots[i].template.power == 0:
                dmgs.append(0.0)
                continue
            dmg = calc_expected_damage(
                me, them, me.move_slots[i].template, self._tc, screens=opp.side,
            )
            dmgs.append(dmg)
            if dmg > best_dmg:
                best_dmg = dmg
                best_idx = i

        for i in range(len(me.move_slots)):
            if scores[i] >= 80 or me.move_slots[i].template.power == 0:
                continue
            if i != best_idx:
                scores[i] += 1

    # ---- AI_Cautious: discourage status after turn 1 ----

    def _layer_cautious(self, scores, me):
        if self._turn <= 1:
            return
        if not self._chance(0.9):
            return
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            if me.move_slots[i].template.power == 0:
                scores[i] += 1

    # ---- AI_Status: status type immunity ----

    def _layer_status(self, scores, me, them):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80:
                continue
            mv = me.move_slots[i].template
            if mv.damage_class != "status" or not mv.meta:
                continue
            aid = mv.meta.get("ailment_id", 0)
            if aid <= 0:
                continue
            status_str = _AILMENT_MAP.get(aid)
            if status_str:
                ok, _ = can_apply_status(them, status_str)
                if not ok:
                    scores[i] += 10

    # ---- AI_Risky: KO detection bonus ----

    def _layer_risky(self, scores, me, them, opp):
        for i in range(len(me.move_slots)):
            if scores[i] >= 80 or me.move_slots[i].template.power == 0:
                continue
            mv = me.move_slots[i].template
            dmg = calc_expected_damage(me, them, mv, self._tc, screens=opp.side)
            if dmg >= them.current_hp:
                scores[i] -= 5
                # self-destruct KO: only 80% of the time at full HP
                if mv.name in _SELF_KO_MOVES and me.hp_frac >= 1.0:
                    if not self._chance(0.8):
                        scores[i] += 5

    # ============================================================
    # SWITCHING
    # ============================================================

    def _should_switch(self, my, opp):
        """Evaluate whether to switch before move selection."""
        me = my.active
        them = opp.active

        alive = [p for i, p in enumerate(my.team)
                 if i != my.active_index and not p.is_fainted]
        if not alive:
            return False

        pressure = 0

        # all my damaging moves resisted
        all_resisted = True
        for slot in me.move_slots:
            if not slot.has_pp or slot.template.power == 0:
                continue
            eff = self._tc.combined_effectiveness(slot.template.type, them.types)
            if eff >= 1.0:
                all_resisted = False
                break
        if all_resisted:
            pressure += 3

        # opponent has super-effective move
        for slot in them.move_slots:
            if slot.template.power == 0:
                continue
            eff = self._tc.combined_effectiveness(slot.template.type, me.types)
            if eff >= 2.0:
                pressure += 2
                break

        # slower and low HP
        if effective_speed(me) < effective_speed(them) and me.hp_frac < 0.4:
            pressure += 2

        if pressure >= 3:
            return self._chance(self._switch_chance)
        return False

    def _pick_switch(self, my, opp) -> Action:
        """Pick the best switch-in based on type matchup."""
        them = opp.active
        best_idx = -1
        best_score = -999.0

        for i, p in enumerate(my.team):
            if i == my.active_index or p.is_fainted:
                continue
            score = 0.0

            # resists opponent's move types
            for slot in them.move_slots:
                if slot.template.power == 0:
                    continue
                eff = self._tc.combined_effectiveness(slot.template.type, p.types)
                if eff <= 0.5:
                    score += 2
                elif eff >= 2.0:
                    score -= 2

            # has super-effective move against opponent
            for slot in p.move_slots:
                if slot.template.power == 0 or not slot.has_pp:
                    continue
                eff = self._tc.combined_effectiveness(slot.template.type, them.types)
                if eff >= 2.0:
                    score += 3
                    break

            # HP-weighted (don't switch to low HP mon)
            score *= p.hp_frac

            # speed bonus
            if effective_speed(p) > effective_speed(them):
                score += 0.5

            # prefer >= 25% HP (Crystal filters below 25%)
            if p.hp_frac < 0.25:
                score -= 5

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            return Switch(team_index=best_idx)

        # fallback: first alive
        for i, p in enumerate(my.team):
            if i != my.active_index and not p.is_fainted:
                return Switch(team_index=i)
        return Struggle()


# ============================================================
# CURATED TRAINER ROSTERS (Pokemon Crystal ROM)
# ============================================================
# Each entry: (species_id, [move_id, ...])
# Movesets from pret/pokecrystal data/trainers/parties.asm
# Moves not in our data substituted with closest valid learnset move

# Lance (Champion) — Gyarados, 3x Dragonite, Aerodactyl, Charizard
TRAINER_LANCE = [
    (130, [175, 240, 57, 63]),   # Gyarados: Flail, Rain Dance, Surf, Hyper Beam
    (149, [86, 239, 87, 63]),    # Dragonite: Thunder Wave, Twister, Thunder, Hyper Beam
    (149, [86, 239, 59, 63]),    # Dragonite: Thunder Wave, Twister, Blizzard, Hyper Beam
    (142, [17, 246, 157, 63]),   # Aerodactyl: Wing Attack, Ancient Power, Rock Slide, Hyper Beam
    (6,   [53, 17, 163, 63]),    # Charizard: Flamethrower, Wing Attack, Slash, Hyper Beam
    (149, [126, 219, 200, 63]),  # Dragonite: Fire Blast, Safeguard, Outrage, Hyper Beam
]

# Karen (E4) — Umbreon, Vileplume, Gengar, Murkrow, Houndoom, Sneasel (pad)
TRAINER_KAREN = [
    (197, [28, 109, 185, 212]),   # Umbreon: Sand Attack, Confuse Ray, Feint Attack, Mean Look
    (45,  [78, 51, 186, 80]),     # Vileplume: Stun Spore, Acid, Moonlight, Petal Dance
    (94,  [122, 180, 174, 194]),  # Gengar: Lick, Spite, Curse, Destiny Bond
    (198, [98, 18, 228, 185]),    # Murkrow: Quick Attack, Whirlwind, Pursuit, Feint Attack
    (229, [46, 228, 53, 242]),    # Houndoom: Roar, Pursuit, Flamethrower, Crunch
    (215, [185, 58, 163, 247]),   # Sneasel (pad): Feint Attack, Ice Beam, Slash, Shadow Ball
]

# Bruno (E4) — Hitmontop, Hitmonlee, Hitmonchan, Onix, Machamp, Heracross (pad)
TRAINER_BRUNO = [
    (237, [228, 98, 91, 197]),    # Hitmontop: Pursuit, Quick Attack, Dig, Detect
    (106, [207, 24, 136, 193]),   # Hitmonlee: Swagger, Double Kick, High Jump Kick, Foresight
    (107, [9, 8, 7, 183]),        # Hitmonchan: Thunder Punch, Ice Punch, Fire Punch, Mach Punch
    (95,  [20, 89, 201, 157]),    # Onix: Bind, Earthquake, Sandstorm, Rock Slide
    (68,  [157, 193, 233, 238]),  # Machamp: Rock Slide, Foresight, Vital Throw, Cross Chop
    (214, [224, 89, 157, 34]),    # Heracross (pad): Megahorn, Earthquake, Rock Slide, Body Slam
]

# Koga (E4) — Ariados, Venomoth, Forretress, Muk, Crobat, Weezing (pad)
TRAINER_KOGA = [
    (168, [104, 169, 226, 202]),  # Ariados: Double Team, Spider Web, Baton Pass, Giga Drain
    (49,  [48, 16, 94, 92]),      # Venomoth: Supersonic, Gust, Psychic, Toxic
    (205, [182, 129, 153, 191]),  # Forretress: Protect, Swift, Explosion, Spikes
    (89,  [107, 151, 188, 92]),   # Muk: Minimize, Acid Armor, Sludge Bomb, Toxic
    (169, [104, 98, 17, 92]),     # Crobat: Double Team, Quick Attack, Wing Attack, Toxic
    (110, [188, 153, 53, 92]),    # Weezing (pad): Sludge Bomb, Explosion, Flamethrower, Toxic
]

# Will (E4) — 2x Xatu, Jynx, Exeggutor, Slowbro, Espeon (pad)
TRAINER_WILL = [
    (178, [98, 248, 109, 94]),    # Xatu: Quick Attack, Future Sight, Confuse Ray, Psychic
    (124, [3, 142, 8, 94]),       # Jynx: Double Slap, Lovely Kiss, Ice Punch, Psychic
    (103, [115, 73, 121, 94]),    # Exeggutor: Reflect, Leech Seed, Egg Bomb, Psychic
    (80,  [174, 133, 34, 94]),    # Slowbro: Curse, Amnesia, Body Slam, Psychic
    (178, [98, 248, 109, 94]),    # Xatu: Quick Attack, Future Sight, Confuse Ray, Psychic
    (196, [94, 234, 129, 247]),   # Espeon (pad): Psychic, Morning Sun, Swift, Shadow Ball
]

# Red (Mt. Silver) — Pikachu, Espeon, Snorlax, Venusaur, Charizard, Blastoise
TRAINER_RED = [
    (25,  [204, 98, 85, 87]),     # Pikachu: Charm, Quick Attack, Thunderbolt, Thunder
    (196, [189, 115, 129, 94]),   # Espeon: Mud-Slap, Reflect, Swift, Psychic
    (143, [133, 173, 156, 34]),   # Snorlax: Amnesia, Snore, Rest, Body Slam
    (3,   [241, 202, 235, 76]),   # Venusaur: Sunny Day, Giga Drain, Synthesis, Solar Beam
    (6,   [53, 17, 163, 83]),     # Charizard: Flamethrower, Wing Attack, Slash, Fire Spin
    (9,   [240, 57, 59, 250]),    # Blastoise: Rain Dance, Surf, Blizzard, Whirlpool
]

# ============================================================
# BOOSTED ROSTERS — same themes, competitive movesets
# ============================================================

BOOSTED_LANCE = [
    (149, [85, 58, 126, 200]),   # Dragonite: Thunderbolt, Ice Beam, Fire Blast, Outrage
    (149, [86, 85, 57, 200]),    # Dragonite: Thunder Wave, Thunderbolt, Surf, Outrage
    (130, [57, 85, 58, 89]),     # Gyarados: Surf, Thunderbolt, Ice Beam, Earthquake
    (142, [89, 157, 17, 126]),   # Aerodactyl: Earthquake, Rock Slide, Wing Attack, Fire Blast
    (6,   [126, 89, 157, 17]),   # Charizard: Fire Blast, Earthquake, Rock Slide, Wing Attack
    (248, [242, 89, 157, 53]),   # Tyranitar: Crunch, Earthquake, Rock Slide, Flamethrower
]

BOOSTED_KAREN = [
    (197, [92, 109, 186, 185]),   # Umbreon: Toxic, Confuse Ray, Moonlight, Feint Attack
    (229, [53, 242, 228, 247]),   # Houndoom: Flamethrower, Crunch, Pursuit, Shadow Ball
    (94,  [247, 85, 188, 95]),    # Gengar: Shadow Ball, Thunderbolt, Sludge Bomb, Hypnosis
    (198, [65, 247, 228, 185]),   # Murkrow: Drill Peck, Shadow Ball, Pursuit, Feint Attack
    (215, [58, 185, 14, 247]),    # Sneasel: Ice Beam, Feint Attack, Swords Dance, Shadow Ball
    (248, [242, 89, 126, 157]),   # Tyranitar: Crunch, Earthquake, Fire Blast, Rock Slide
]

BOOSTED_BRUNO = [
    (68,  [238, 89, 157, 53]),    # Machamp: Cross Chop, Earthquake, Rock Slide, Flamethrower
    (214, [224, 89, 14, 157]),    # Heracross: Megahorn, Earthquake, Swords Dance, Rock Slide
    (107, [9, 8, 7, 183]),        # Hitmonchan: Thunder Punch, Ice Punch, Fire Punch, Mach Punch
    (106, [136, 89, 157, 34]),    # Hitmonlee: High Jump Kick, Earthquake, Rock Slide, Body Slam
    (208, [89, 231, 157, 153]),   # Steelix: Earthquake, Iron Tail, Rock Slide, Explosion
    (237, [89, 229, 228, 98]),    # Hitmontop: Earthquake, Rapid Spin, Pursuit, Quick Attack
]

BOOSTED_KOGA = [
    (169, [188, 247, 109, 17]),   # Crobat: Sludge Bomb, Shadow Ball, Confuse Ray, Wing Attack
    (94,  [247, 85, 188, 95]),    # Gengar: Shadow Ball, Thunderbolt, Sludge Bomb, Hypnosis
    (89,  [188, 53, 85, 153]),    # Muk: Sludge Bomb, Flamethrower, Thunderbolt, Explosion
    (205, [191, 229, 92, 153]),   # Forretress: Spikes, Rapid Spin, Toxic, Explosion
    (110, [188, 53, 85, 153]),    # Weezing: Sludge Bomb, Flamethrower, Thunderbolt, Explosion
    (34,  [89, 188, 58, 85]),     # Nidoking: Earthquake, Sludge Bomb, Ice Beam, Thunderbolt
]

BOOSTED_WILL = [
    (121, [94, 85, 58, 105]),     # Starmie: Psychic, Thunderbolt, Ice Beam, Recover
    (124, [94, 58, 142, 247]),    # Jynx: Psychic, Ice Beam, Lovely Kiss, Shadow Ball
    (103, [94, 202, 95, 153]),    # Exeggutor: Psychic, Giga Drain, Hypnosis, Explosion
    (80,  [94, 57, 58, 86]),      # Slowbro: Psychic, Surf, Ice Beam, Thunder Wave
    (196, [94, 234, 247, 237]),   # Espeon: Psychic, Morning Sun, Shadow Ball, Hidden Power
    (178, [94, 247, 86, 115]),    # Xatu: Psychic, Shadow Ball, Thunder Wave, Reflect
]

BOOSTED_RED = [
    (25,  [85, 86, 237, 231]),    # Pikachu: Thunderbolt, Thunder Wave, Hidden Power, Iron Tail
    (196, [94, 234, 247, 237]),   # Espeon: Psychic, Morning Sun, Shadow Ball, Hidden Power
    (143, [174, 34, 156, 214]),   # Snorlax: Curse, Body Slam, Rest, Sleep Talk
    (3,   [79, 73, 202, 188]),    # Venusaur: Sleep Powder, Leech Seed, Giga Drain, Sludge Bomb
    (6,   [126, 89, 157, 17]),    # Charizard: Fire Blast, Earthquake, Rock Slide, Wing Attack
    (9,   [57, 58, 229, 237]),    # Blastoise: Surf, Ice Beam, Rapid Spin, Hidden Power
]

# all trainer rosters keyed by name
TRAINERS: dict[str, list[tuple[int, list[int]]]] = {
    "lance": TRAINER_LANCE,
    "karen": TRAINER_KAREN,
    "bruno": TRAINER_BRUNO,
    "koga": TRAINER_KOGA,
    "will": TRAINER_WILL,
    "red": TRAINER_RED,
}

BOOSTED_TRAINERS: dict[str, list[tuple[int, list[int]]]] = {
    "lance": BOOSTED_LANCE,
    "karen": BOOSTED_KAREN,
    "bruno": BOOSTED_BRUNO,
    "koga": BOOSTED_KOGA,
    "will": BOOSTED_WILL,
    "red": BOOSTED_RED,
}

ALL_TRAINER_NAMES = list(TRAINERS.keys())


def build_trainer_team(
    data: "DataStore", trainer: str, rng: random.Random | None = None,
    boosted: bool = False,
) -> list[Pokemon]:
    """Build a curated 6-mon team for a Crystal ROM trainer.

    boosted=True uses competitive movesets that stay on-theme.
    """
    from gym_env.team_builder import _make_pokemon

    source = BOOSTED_TRAINERS if boosted else TRAINERS
    roster = source[trainer]
    team = [_make_pokemon(data, sid, mids) for sid, mids in roster]
    if rng:
        rng.shuffle(team)
    return team
