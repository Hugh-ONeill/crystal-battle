# team building: curated templates, role-based composition, and random fallback
# Usage: build_team(data, rng=rng, tier="ou") for the 50/35/15 dispatcher

from __future__ import annotations

import random as _random

from engine.data_loader import DataStore
from engine.move import MoveTemplate, MoveSlot
from engine.pokemon import Pokemon, PokemonSpecies

# ailment_ids in v2 scope
STATUS_AILMENTS = {1, 2, 3, 4, 5, 6}

# Gen 2 physical/special split is by type, not per-move
PHYSICAL_TYPES = {"normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel"}
SPECIAL_TYPES = {"fire", "water", "electric", "grass", "ice", "psychic", "dragon", "dark"}

# self-KO moves -- too situational for auto-pick
SELF_KO_MOVES = {120, 153}  # Self-Destruct, Explosion

# Dream Eater only works if target is asleep -- useless without a sleep move
_DREAM_EATER = 138
_SLEEP_MOVES = {95, 79, 147, 142, 47}  # Hypnosis, Sleep Powder, Spore, Lovely Kiss, Sing

# multi-turn moves: effective power should account for wasted turns
_RECHARGE_MOVES = {63}                          # Hyper Beam (150 / 2 turns = 75 effective)
_CHARGE_MOVES = {76, 13, 130, 143}              # Solar Beam, Razor Wind, Skull Bash, Sky Attack
_CHARGE_INVULN = {19, 91}                       # Fly, Dig (semi-invulnerable charge)
_LOCKIN_MOVES = {37, 80, 200}                   # Thrash, Petal Dance, Outrage (lock-in + confusion)
_MULTI_TURN_PENALTY = _RECHARGE_MOVES | _CHARGE_MOVES | _CHARGE_INVULN | _LOCKIN_MOVES

# junk moves: too weak, strictly worse than alternatives, or useless without items
_JUNK_MOVES = {
    # strictly worse versions of common moves
    60,   # Psybeam (65) -- worse Psychic (90)
    93,   # Confusion (50) -- worse Psychic
    209,  # Spark (65) -- worse Thunderbolt (95)
    84,   # Thunder Shock (40) -- worse Thunderbolt
    52,   # Ember (40) -- worse Flamethrower (90)
    55,   # Water Gun (40) -- worse Surf (90)
    181,  # Powder Snow (40) -- worse Ice Beam (90)
    16,   # Gust (40) -- worse Drill Peck/Wing Attack
    # very low power / useless
    189,  # Mud-Slap (20)
    132,  # Constrict (10)
    122,  # Lick (20)
    # 180 = Spite -- real effect (PP drain) but rarely worth a slot
    250,  # Whirlpool (15)
    20,   # Bind (15)
    210,  # Fury Cutter (10 base, unreliable ramp)
    249,  # Rock Smash (20)
    70,   # Strength (80 normal, just filler)
    15,   # Cut (50 normal, HM filler)
    # unreliable / no items
    168,  # Thief (40, items not implemented)
    248,  # Future Sight (delayed, can't crit, ignores type)
    36,   # Take Down (90) -- worse Double-Edge (120)
    5,    # Mega Punch (80) -- worse Return (102)
    146,  # Struggle -- never pick this
}


# ============================================================
# SMOGON GSC TIERS
# ============================================================

# source: smogon.com/dex/gs/ -- pokemon not listed default to NU
UBERS = {
    150, 151, 249, 250, 251,  # Mewtwo, Mew, Lugia, Ho-Oh, Celebi
}

OU = {
    3,    # Venusaur
    6,    # Charizard
    34,   # Nidoking
    36,   # Clefable
    59,   # Arcanine
    65,   # Alakazam
    68,   # Machamp
    76,   # Golem
    89,   # Muk
    91,   # Cloyster
    94,   # Gengar
    103,  # Exeggutor
    105,  # Marowak
    112,  # Rhydon
    121,  # Starmie
    124,  # Jynx
    125,  # Electabuzz
    127,  # Pinsir
    130,  # Gyarados
    131,  # Lapras
    134,  # Vaporeon
    135,  # Jolteon
    139,  # Omastar
    141,  # Kabutops
    142,  # Aerodactyl
    143,  # Snorlax
    145,  # Zapdos
    149,  # Dragonite
    196,  # Espeon
    197,  # Umbreon
    205,  # Forretress
    208,  # Steelix
    212,  # Scizor
    214,  # Heracross
    227,  # Skarmory
    241,  # Miltank
    242,  # Blissey
    243,  # Raikou
    245,  # Suicune
    248,  # Tyranitar
}

UU = {
    9,    # Blastoise
    26,   # Raichu
    28,   # Sandslash
    31,   # Nidoqueen
    38,   # Ninetales
    45,   # Vileplume
    51,   # Dugtrio
    55,   # Golduck
    62,   # Poliwrath
    71,   # Victreebel
    73,   # Tentacruel
    80,   # Slowbro
    82,   # Magneton
    85,   # Dodrio
    97,   # Hypno
    101,  # Electrode
    110,  # Weezing
    113,  # Chansey
    115,  # Kangaskhan
    122,  # Mr. Mime
    123,  # Scyther
    126,  # Magmar
    128,  # Tauros
    154,  # Meganium
    157,  # Typhlosion
    160,  # Feraligatr
    169,  # Crobat
    171,  # Lanturn
    181,  # Ampharos
    182,  # Bellossom
    186,  # Politoed
    189,  # Jumpluff
    195,  # Quagsire
    199,  # Slowking
    203,  # Girafarig
    210,  # Granbull
    211,  # Qwilfish
    217,  # Ursaring
    229,  # Houndoom
    230,  # Kingdra
    232,  # Donphan
    233,  # Porygon2
    237,  # Hitmontop
    244,  # Entei
}


def get_tier(species_id: int) -> str:
    """Return Smogon GSC tier for a pokemon."""
    if species_id in UBERS:
        return "uber"
    if species_id in OU:
        return "ou"
    if species_id in UU:
        return "uu"
    return "nu"


# ============================================================
# TEAM TEMPLATES
# ============================================================

# 16 hand-crafted teams: (species_id, [move_ids])
# all moves verified against data/*.json learnsets
TEMPLATES = [
    [  # 0: Paralysis Spread -- TW the field, sweep with speed advantage
        (145, [86, 85, 65, 237]),   # Zapdos: TW, Tbolt, Drill Peck, HP
        (149, [86, 58, 126, 57]),   # Dragonite: TW, Ice Beam, Fire Blast, Surf
        (206, [137, 85, 157, 216]), # Dunsparce: Glare, Tbolt, Rock Slide, Return
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (143, [34, 89, 156, 58]),   # Snorlax: Body Slam, EQ, Rest, Ice Beam
    ],
    [  # 1: Sleep-and-Sweep -- Spore + Hypnosis for 2-pokemon sleep advantage
        (47, [147, 202, 188, 237]),  # Parasect: Spore, Giga Drain, Sludge Bomb, HP
        (94, [95, 247, 85, 188]),    # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
        (103, [95, 94, 202, 188]),   # Exeggutor: Hypnosis, Psychic, Giga Drain, Sludge Bomb
        (214, [224, 89, 216, 237]),  # Heracross: Megahorn, EQ, Return, HP
        (248, [242, 89, 157, 126]),  # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (121, [57, 94, 85, 58]),     # Starmie: Surf, Psychic, Tbolt, Ice Beam
    ],
    [  # 2: Toxic Stall -- wall up, toxic everything
        (197, [92, 212, 109, 156]), # Umbreon: Toxic, Mean Look, Confuse Ray, Rest
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (245, [92, 57, 58, 156]),   # Suicune: Toxic, Surf, Ice Beam, Rest
        (208, [92, 89, 231, 156]),  # Steelix: Toxic, EQ, Iron Tail, Rest
        (205, [92, 191, 237, 156]), # Forretress: Toxic, Spikes, HP, Rest
        (91, [92, 191, 57, 58]),    # Cloyster: Toxic, Spikes, Surf, Ice Beam
    ],
    [  # 3: Ho-Oh Offense -- Sacred Fire burns cripple physicals (ubers)
        (250, [221, 89, 105, 237]), # Ho-Oh: Sacred Fire, EQ, Recover, HP
        (150, [94, 58, 85, 126]),   # Mewtwo: Psychic, Ice Beam, Tbolt, Fire Blast
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (149, [58, 126, 85, 57]),   # Dragonite: Ice Beam, Fire Blast, Tbolt, Surf
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
    ],
    [  # 4: Ghost Disruption -- Hypnosis + Confuse Ray, Normal/Fighting immunity
        (94, [95, 247, 85, 109]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Confuse Ray
        (65, [94, 247, 86, 105]),   # Alakazam: Psychic, Shadow Ball, TW, Recover
        (131, [57, 58, 85, 109]),   # Lapras: Surf, Ice Beam, Tbolt, Confuse Ray
        (197, [92, 109, 156, 237]), # Umbreon: Toxic, Confuse Ray, Rest, HP
        (214, [224, 89, 216, 237]), # Heracross: Megahorn, EQ, Return, HP
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
    ],
    [  # 5: Jynx Psychic Sweep -- Lovely Kiss lead into triple Psychic coverage
        (124, [142, 94, 58, 237]),  # Jynx: Lovely Kiss, Psychic, Ice Beam, HP
        (65, [94, 247, 86, 105]),   # Alakazam: Psychic, Shadow Ball, TW, Recover
        (196, [94, 237, 234, 247]), # Espeon: Psychic, HP, Morning Sun, Shadow Ball
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (143, [34, 89, 156, 58]),   # Snorlax: Body Slam, EQ, Rest, Ice Beam
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
    ],
    [  # 6: Physical Blitz -- brute force + Arbok Glare support
        (214, [224, 89, 216, 237]), # Heracross: Megahorn, EQ, Return, HP
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (24, [137, 188, 89, 216]),  # Arbok: Glare, Sludge Bomb, EQ, Return
        (112, [89, 58, 126, 231]),  # Rhydon: EQ, Ice Beam, Fire Blast, Iron Tail
        (217, [216, 89, 14, 157]),  # Ursaring: Return, EQ, Swords Dance, Rock Slide
    ],
    [  # 7: Venusaur Sleep Stall -- Sleep Powder + Spore + Toxic + Spikes attrition
        (3, [79, 73, 202, 237]),    # Venusaur: Sleep Powder, Leech Seed, Giga Drain, HP
        (47, [147, 202, 188, 237]), # Parasect: Spore, Giga Drain, Sludge Bomb, HP
        (205, [92, 191, 237, 156]), # Forretress: Toxic, Spikes, HP, Rest
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (245, [92, 57, 58, 156]),   # Suicune: Toxic, Surf, Ice Beam, Rest
        (208, [92, 89, 231, 156]),  # Steelix: Toxic, EQ, Iron Tail, Rest
    ],
    [  # 8: CurseLax -- Curse + Rest/Talk Snorlax behind Spikes + screens
        (143, [174, 156, 214, 34]), # Snorlax: Curse, Rest, Sleep Talk, Body Slam
        (227, [191, 65, 174, 156]), # Skarmory: Spikes, Drill Peck, Curse, Rest
        (242, [113, 135, 58, 85]),  # Blissey: Light Screen, Soft-Boiled, Ice Beam, Tbolt
        (121, [115, 229, 57, 94]),  # Starmie: Reflect, Rapid Spin, Surf, Psychic
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (214, [224, 89, 216, 237]), # Heracross: Megahorn, EQ, Return, HP
    ],
    [  # 9: Rain Dance -- Thunder + boosted Surfs
        (245, [240, 57, 58, 156]),  # Suicune: Rain Dance, Surf, Ice Beam, Rest
        (145, [87, 65, 240, 237]),  # Zapdos: Thunder, Drill Peck, Rain Dance, HP
        (121, [240, 57, 85, 94]),   # Starmie: Rain Dance, Surf, Tbolt, Psychic
        (134, [240, 57, 58, 156]),  # Vaporeon: Rain Dance, Surf, Ice Beam, Rest
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (94, [95, 247, 85, 188]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
    ],
    [  # 10: Spikes Stacking -- dual Spiker + Rapid Spin, force switches
        (205, [191, 92, 237, 156]), # Forretress: Spikes, Toxic, HP, Rest
        (91, [191, 57, 58, 92]),    # Cloyster: Spikes, Surf, Ice Beam, Toxic
        (121, [229, 57, 94, 85]),   # Starmie: Rapid Spin, Surf, Psychic, Tbolt
        (227, [18, 65, 92, 156]),   # Skarmory: Whirlwind, Drill Peck, Toxic, Rest
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
    ],
    [  # 11: SD Sweepers -- Swords Dance + coverage
        (212, [14, 232, 237, 216]), # Scizor: Swords Dance, Metal Claw, HP, Return
        (105, [14, 155, 157, 89]),  # Marowak: Swords Dance, Bonemerang, Rock Slide, EQ
        (214, [14, 224, 89, 216]),  # Heracross: Swords Dance, Megahorn, EQ, Return
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (145, [86, 85, 65, 237]),   # Zapdos: TW, Tbolt, Drill Peck, HP
    ],
    [  # 12: Baton Pass Chain -- Girafarig passes Agility/Amnesia to sweepers
        (203, [226, 97, 133, 94]),  # Girafarig: Baton Pass, Agility, Amnesia, Psychic
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (214, [224, 89, 216, 237]), # Heracross: Megahorn, EQ, Return, HP
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
    ],
    [  # 13: CurseMilk -- Miltank Curse tank + status spread
        (241, [174, 34, 89, 156]),  # Miltank: Curse, Body Slam, EQ, Rest
        (94, [95, 247, 85, 188]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
        (145, [86, 85, 65, 237]),   # Zapdos: TW, Tbolt, Drill Peck, HP
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (205, [191, 92, 237, 156]), # Forretress: Spikes, Toxic, HP, Rest
    ],
    [  # 14: Raikou Offense -- fast special sweeper + support
        (243, [85, 242, 115, 156]), # Raikou: Tbolt, Crunch, Reflect, Rest
        (94, [95, 247, 85, 188]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
        (149, [86, 58, 126, 57]),   # Dragonite: TW, Ice Beam, Fire Blast, Surf
        (91, [191, 57, 58, 92]),    # Cloyster: Spikes, Surf, Ice Beam, Toxic
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (197, [92, 109, 156, 237]), # Umbreon: Toxic, Confuse Ray, Rest, HP
    ],
    [  # 15: Nidoking Mixed -- wide coverage + status support
        (34, [89, 58, 85, 126]),    # Nidoking: EQ, Ice Beam, Tbolt, Fire Blast
        (94, [95, 247, 85, 188]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
        (143, [174, 156, 214, 34]), # Snorlax: Curse, Rest, Sleep Talk, Body Slam
        (227, [191, 65, 92, 156]),  # Skarmory: Spikes, Drill Peck, Toxic, Rest
        (245, [92, 57, 58, 156]),   # Suicune: Toxic, Surf, Ice Beam, Rest
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
    ],
    [  # 16: SD Marowak Sweep -- para support into SD Marowak cleanup
        (105, [14, 155, 89, 157]),  # Marowak: Swords Dance, Bonemerang, EQ, Rock Slide
        (145, [86, 85, 65, 237]),   # Zapdos: TW, Tbolt, Drill Peck, HP
        (149, [86, 58, 126, 57]),   # Dragonite: TW, Ice Beam, Fire Blast, Surf
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (227, [191, 65, 92, 156]),  # Skarmory: Spikes, Drill Peck, Toxic, Rest
    ],
    [  # 17: Growth Vaporeon -- Growth behind screens + bulky waters
        (134, [74, 57, 58, 156]),   # Vaporeon: Growth, Surf, Ice Beam, Rest
        (245, [57, 58, 115, 156]),  # Suicune: Surf, Ice Beam, Reflect, Rest
        (242, [113, 135, 58, 85]),  # Blissey: Light Screen, Soft-Boiled, Ice Beam, Tbolt
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (94, [95, 247, 85, 188]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
        (121, [57, 94, 85, 58]),    # Starmie: Surf, Psychic, Tbolt, Ice Beam
    ],
    [  # 18: Double Curse -- CurseLax + CurseSteelix behind Spikes
        (143, [174, 156, 214, 34]), # Snorlax: Curse, Rest, Sleep Talk, Body Slam
        (208, [174, 89, 231, 156]), # Steelix: Curse, EQ, Iron Tail, Rest
        (205, [191, 92, 237, 156]), # Forretress: Spikes, Toxic, HP, Rest
        (121, [229, 57, 94, 85]),   # Starmie: Rapid Spin, Surf, Psychic, Tbolt
        (94, [95, 247, 85, 188]),   # Gengar: Hypnosis, Shadow Ball, Tbolt, Sludge Bomb
        (214, [224, 89, 216, 237]), # Heracross: Megahorn, EQ, Return, HP
    ],
    [  # 20: Minimize Stall -- Clefable + Muk evasion stall behind Spikes
        (36,  [107, 135, 58, 126]),   # Clefable: Minimize, Soft-Boiled, Ice Beam, Fire Blast
        (89,  [107, 188, 126, 156]),  # Muk: Minimize, Sludge Bomb, Fire Blast, Rest
        (205, [191, 92, 237, 156]),   # Forretress: Spikes, Toxic, HP, Rest
        (121, [57, 94, 85, 58]),      # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (248, [242, 89, 157, 126]),   # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (145, [86, 85, 65, 237]),     # Zapdos: TW, Tbolt, Drill Peck, HP
    ],
    [  # 19: Agility Sweep -- Agility + coverage for late-game cleanup
        (196, [97, 94, 237, 247]),  # Espeon: Agility, Psychic, HP, Shadow Ball
        (135, [97, 85, 237, 58]),   # Jolteon: Agility, Tbolt, HP, Ice Beam
        (68, [238, 89, 157, 126]),  # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (242, [92, 135, 58, 85]),   # Blissey: Toxic, Soft-Boiled, Ice Beam, Tbolt
        (248, [242, 89, 157, 126]), # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (227, [191, 65, 92, 156]),  # Skarmory: Spikes, Drill Peck, Toxic, Rest
    ],
]


# ============================================================
# OFFENSIVE TEMPLATES (4 damaging moves per mon, pure coverage)
# ============================================================

OFFENSIVE_TEMPLATES = [
    [  # 0: Mixed Offense -- balanced phys/spec attackers with wide coverage
        (248, [242, 89, 157, 126]),  # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (121, [57, 94, 85, 58]),     # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (68, [238, 89, 157, 126]),   # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (94, [247, 85, 188, 58]),    # Gengar: Shadow Ball, Tbolt, Sludge Bomb, Ice Beam
        (214, [224, 89, 216, 157]),  # Heracross: Megahorn, EQ, Return, Rock Slide
        (149, [58, 126, 85, 57]),    # Dragonite: Ice Beam, Fire Blast, Tbolt, Surf
    ],
    [  # 1: Special Blitz -- fast special sweepers
        (65, [94, 58, 85, 247]),     # Alakazam: Psychic, Ice Beam, Tbolt, Shadow Ball
        (121, [57, 94, 85, 58]),     # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (94, [247, 85, 188, 58]),    # Gengar: Shadow Ball, Tbolt, Sludge Bomb, Ice Beam
        (145, [85, 65, 237, 58]),    # Zapdos: Tbolt, Drill Peck, HP, Ice Beam (via event)
        (196, [94, 237, 247, 58]),   # Espeon: Psychic, HP, Shadow Ball, Ice Beam (coverage)
        (34, [89, 58, 85, 126]),     # Nidoking: EQ, Ice Beam, Tbolt, Fire Blast
    ],
    [  # 2: Physical Blitz -- raw physical power
        (214, [224, 89, 216, 157]),  # Heracross: Megahorn, EQ, Return, Rock Slide
        (68, [238, 89, 157, 126]),   # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (248, [242, 89, 157, 126]),  # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (112, [89, 58, 157, 231]),   # Rhydon: EQ, Ice Beam, Rock Slide, Iron Tail
        (217, [216, 89, 157, 126]),  # Ursaring: Return, EQ, Rock Slide, Fire Blast
        (130, [57, 216, 89, 58]),    # Gyarados: Surf, Return, EQ, Ice Beam
    ],
    [  # 3: Speed Offense -- fast attackers that outspeed and 2HKO
        (121, [57, 94, 85, 58]),     # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (65, [94, 58, 85, 247]),     # Alakazam: Psychic, Ice Beam, Tbolt, Shadow Ball
        (135, [85, 237, 58, 247]),   # Jolteon: Tbolt, HP, Ice Beam (coverage), Shadow Ball
        (142, [216, 89, 157, 58]),   # Aerodactyl: Return, EQ, Rock Slide, Ice Beam
        (94, [247, 85, 188, 58]),    # Gengar: Shadow Ball, Tbolt, Sludge Bomb, Ice Beam
        (124, [94, 58, 237, 188]),   # Jynx: Psychic, Ice Beam, HP, Sludge Bomb
    ],
    [  # 4: Dragon Core -- Dragonite + coverage partners
        (149, [58, 126, 85, 57]),    # Dragonite: Ice Beam, Fire Blast, Tbolt, Surf
        (248, [242, 89, 157, 126]),  # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (121, [57, 94, 85, 58]),     # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (214, [224, 89, 216, 157]),  # Heracross: Megahorn, EQ, Return, Rock Slide
        (34, [89, 58, 85, 126]),     # Nidoking: EQ, Ice Beam, Tbolt, Fire Blast
        (94, [247, 85, 188, 58]),    # Gengar: Shadow Ball, Tbolt, Sludge Bomb, Ice Beam
    ],
    [  # 5: Anti-Stall Offense -- broad coverage to break walls
        (68, [238, 89, 157, 126]),   # Machamp: Cross Chop, EQ, Rock Slide, Fire Blast
        (34, [89, 58, 85, 126]),     # Nidoking: EQ, Ice Beam, Tbolt, Fire Blast
        (248, [242, 89, 157, 126]),  # Tyranitar: Crunch, EQ, Rock Slide, Fire Blast
        (130, [57, 216, 89, 58]),    # Gyarados: Surf, Return, EQ, Ice Beam
        (121, [57, 94, 85, 58]),     # Starmie: Surf, Psychic, Tbolt, Ice Beam
        (149, [58, 126, 85, 57]),    # Dragonite: Ice Beam, Fire Blast, Tbolt, Surf
    ],
]


# ============================================================
# ROLE-BASED POOLS
# ============================================================

STATUS_LEAD_IDS = [
    145, 149, 206, 82, 181, 135, 24, 47, 103, 124, 94, 3, 36, 131,
]
PHYS_SWEEPER_IDS = [68, 214, 248, 149, 112, 123, 127, 217, 232, 212]
SPEC_SWEEPER_IDS = [65, 94, 196, 121, 150, 145, 181, 229, 103, 124]
PHYS_WALL_IDS = [208, 205, 91, 95, 76, 213]
SPEC_WALL_IDS = [242, 143, 197, 245, 226, 249]

# status move priority: Spore > Lovely Kiss > Glare > TW > Sleep Powder > Stun Spore > Hypnosis
STATUS_MOVE_PRIORITY = [147, 142, 137, 86, 79, 78, 95]

# recovery priority: Soft-Boiled > Recover > Morning Sun > Synthesis > Rest
RECOVERY_PRIORITY = [135, 105, 234, 235, 156]

# setup move priority: Swords Dance > Curse > Agility > Amnesia > Growth > Meditate
SETUP_PRIORITY = [14, 174, 97, 133, 74, 96]


# ============================================================
# HELPERS
# ============================================================

def _get_status_moves(data: DataStore, species_id: int) -> list[int]:
    """Return move ids from learnset that apply a v2-scoped status."""
    pkmn = data.pokemon[species_id]
    result = []
    for mid in pkmn["learnset"]:
        mdata = data.moves.get(mid)
        if mdata is None:
            continue
        if mdata["damage_class"] != "status":
            continue
        meta = mdata.get("meta")
        if meta and meta.get("ailment_id", 0) in STATUS_AILMENTS:
            result.append(mid)
    return result


def _sanitize_moves(data: DataStore, species_id: int, move_ids: list[int]) -> list[int]:
    """Remove moves that don't make sense without prerequisites.

    Dream Eater is replaced if no sleep move is in the moveset.
    """
    if _DREAM_EATER not in move_ids:
        return move_ids
    if set(move_ids) & _SLEEP_MOVES:
        return move_ids
    result = [m for m in move_ids if m != _DREAM_EATER]
    exclude = set(result) | {_DREAM_EATER}
    replacement = _pick_coverage_moves_with_exclude(data, species_id, 1, exclude)
    result.extend(replacement)
    return result[:4]


def _make_pokemon(data: DataStore, species_id: int, move_ids: list[int],
                  hp_type: str | None = None) -> Pokemon:
    """Build a Pokemon instance from species id and move ids.

    hp_type: desired Hidden Power type. If None and HP is in moves,
    picks a coverage type based on the mon's types.
    """
    from engine.pokemon import dvs_for_hidden_power
    move_ids = _sanitize_moves(data, species_id, move_ids)
    pdata = data.pokemon[species_id]
    species = PokemonSpecies.from_dict(pdata)
    templates = []
    for mid in move_ids:
        mdata = data.moves.get(mid)
        if mdata:
            templates.append(MoveTemplate.from_dict(mdata))

    # set DVs for desired HP type
    has_hp = 237 in move_ids
    dvs = {}
    if has_hp:
        if hp_type is None:
            hp_type = _pick_hp_coverage_type(pdata["types"])
        dvs = dvs_for_hidden_power(hp_type)

    return Pokemon(species=species, dvs=dvs, move_slots=[MoveSlot(template=t) for t in templates])


# preferred HP types for coverage based on the mon's own types
_HP_COVERAGE = {
    "electric": "ice",       # BoltBeam coverage
    "water": "electric",     # hit other waters
    "fire": "grass",         # hit water/ground
    "grass": "fire",         # hit steel/bug
    "ice": "electric",       # hit water
    "psychic": "fire",       # hit steel/dark
    "bug": "rock",           # hit flying/fire
    "fighting": "ghost",     # hit psychic
    "normal": "fighting",    # hit rock/steel
    "ground": "ice",         # hit flying/grass
    "rock": "ground",        # hit electric/fire
    "steel": "fire",         # hit other steel
    "dark": "fighting",      # hit other dark
    "ghost": "dark",         # hit other ghost
    "flying": "ground",      # hit electric/rock
    "poison": "ground",      # hit other poison
    "dragon": "ice",         # hit other dragon
}

def _pick_hp_coverage_type(mon_types: list[str]) -> str:
    """Pick a Hidden Power type that gives coverage against the mon's weaknesses."""
    for t in mon_types:
        if t in _HP_COVERAGE:
            return _HP_COVERAGE[t]
    return "ice"  # default to ice (most broadly useful)


def _pick_status_move(data: DataStore, species_id: int) -> int | None:
    """Pick the best status move from priority list, falling back to any status move."""
    learnset = set(data.pokemon[species_id]["learnset"])
    for mid in STATUS_MOVE_PRIORITY:
        if mid in learnset:
            return mid
    status = _get_status_moves(data, species_id)
    return status[0] if status else None


def _effective_power(data: DataStore, species_id: int, move_id: int) -> float:
    """Estimate a move's effective power accounting for Gen 2 phys/spec split
    and multi-turn move penalties.

    Scales raw power by the ratio of the matching attack stat to the mon's
    better stat, so physical moves on special attackers get penalized.
    Recharge/charge moves are halved since they take 2 turns.
    """
    m = data.moves[move_id]
    power = m.get("power", 0)
    if power == 0:
        return 0.0
    bs = data.pokemon[species_id]["base_stats"]
    atk, spa = bs["attack"], bs["special_attack"]
    mtype = m.get("type", "normal")
    is_phys = mtype in PHYSICAL_TYPES
    stat = atk if is_phys else spa
    best_stat = max(atk, spa)
    ratio = stat / best_stat if best_stat > 0 else 1.0
    eff = power * ratio
    # STAB: 1.5x for moves matching the pokemon's type
    types = data.pokemon[species_id]["types"]
    if mtype in types:
        eff *= 1.5
    # accuracy penalty: scale by hit rate
    acc = m.get("accuracy")
    if acc is not None and acc < 100:
        eff *= acc / 100.0
    # multi-turn moves: halve effective power (2 turns per use)
    if move_id in _MULTI_TURN_PENALTY:
        eff *= 0.5
    return eff


def _pick_best_moves(
    data: DataStore, species_id: int, count: int, exclude: set[int] | None = None,
) -> list[int]:
    """Pick the N strongest damaging moves by effective power, excluding specified IDs."""
    if exclude is None:
        exclude = set()
    # exclude Dream Eater unless the mon has a sleep move in its chosen moveset
    learnset = set(data.pokemon[species_id]["learnset"])
    has_sleep = bool(learnset & _SLEEP_MOVES)
    skip = SELF_KO_MOVES | _MULTI_TURN_PENALTY | _JUNK_MOVES
    if not has_sleep:
        skip = skip | {_DREAM_EATER}
    damaging = [m for m in data.get_damaging_moves(species_id)
                if m not in exclude and m not in skip]
    damaging.sort(key=lambda m: _effective_power(data, species_id, m), reverse=True)
    result = damaging[:count]
    if len(result) < count:
        used = exclude | set(result)
        remaining = [m for m in data.pokemon[species_id]["learnset"] if m not in used]
        result.extend(remaining[:count - len(result)])
    return result[:count]


def _pick_recovery_move(data: DataStore, species_id: int) -> int | None:
    """Pick the best recovery move from priority list."""
    learnset = set(data.pokemon[species_id]["learnset"])
    for mid in RECOVERY_PRIORITY:
        if mid in learnset:
            return mid
    return None


def _pick_setup_move(data: DataStore, species_id: int) -> int | None:
    """Pick the best setup/boost move from priority list."""
    learnset = set(data.pokemon[species_id]["learnset"])
    for mid in SETUP_PRIORITY:
        if mid in learnset:
            return mid
    return None


def _pick_wall_moves(data: DataStore, species_id: int) -> list[int]:
    """Pick wall moves: Toxic + recovery + strongest damaging to fill 4 slots."""
    learnset = set(data.pokemon[species_id]["learnset"])
    moves: list[int] = []
    exclude: set[int] = set()
    if 92 in learnset:  # toxic
        moves.append(92)
        exclude.add(92)
    recovery = _pick_recovery_move(data, species_id)
    if recovery is not None:
        moves.append(recovery)
        exclude.add(recovery)
    need = 4 - len(moves)
    moves.extend(_pick_best_moves(data, species_id, need, exclude))
    return moves[:4]


def _type_ok(type_counts: dict[str, int], new_types: list[str], max_shared: int = 2) -> bool:
    """Check if adding a pokemon with new_types keeps all type counts <= max_shared."""
    for t in new_types:
        if type_counts.get(t, 0) >= max_shared:
            return False
    return True


# ============================================================
# BUILDERS
# ============================================================

def build_random_team(
    data: DataStore,
    team_size: int = 6,
    rng: _random.Random | None = None,
    allowed_tiers: set[str] | None = None,
) -> list[Pokemon]:
    """Build a random team of pokemon with 3 damaging + 1 status move when possible."""
    if rng is None:
        rng = _random.Random()

    # filter to pokemon with at least 1 damaging move
    eligible = []
    for sid, pdata in data.pokemon.items():
        if allowed_tiers and get_tier(sid) not in allowed_tiers:
            continue
        damaging = data.get_damaging_moves(sid)
        if len(damaging) >= 1:
            eligible.append(sid)

    chosen_ids = rng.sample(eligible, min(team_size, len(eligible)))
    team = []

    for sid in chosen_ids:
        pdata = data.pokemon[sid]
        species = PokemonSpecies.from_dict(pdata)

        status = _get_status_moves(data, sid)
        setup_mid = _pick_setup_move(data, sid)

        # decide the utility slot first so we know if sleep is available
        utility_mid = None
        if status and len(data.get_damaging_moves(sid)) >= 3:
            status_mid = _pick_status_move(data, sid) or rng.choice(status)
            if setup_mid and rng.random() < 0.3:
                utility_mid = setup_mid
            else:
                utility_mid = status_mid
        elif setup_mid and len(data.get_damaging_moves(sid)) >= 3:
            utility_mid = setup_mid

        # pick damaging moves with type-diversity, excluding Dream Eater if no sleep
        has_sleep = utility_mid in _SLEEP_MOVES if utility_mid else False
        exclude_dmg = {utility_mid} if utility_mid else set()
        if not has_sleep:
            exclude_dmg.add(_DREAM_EATER)
        chosen_moves = _pick_coverage_moves(data, sid,
                            3 if utility_mid else 4,
                            exclude=exclude_dmg, has_sleep_override=has_sleep)
        if utility_mid:
            chosen_moves.append(utility_mid)

        move_templates = []
        for mid in chosen_moves:
            mdata = data.moves[mid]
            move_templates.append(MoveTemplate.from_dict(mdata))

        pkmn = Pokemon.from_species(species, move_templates)
        team.append(pkmn)

    return team


def _build_from_template(data: DataStore, rng: _random.Random) -> list[Pokemon]:
    """Build a team from a random pre-defined template."""
    template = rng.choice(TEMPLATES)
    return [_make_pokemon(data, sid, mids) for sid, mids in template]


def _build_role_based(data: DataStore, rng: _random.Random,
                      allowed_tiers: set[str] | None = None) -> list[Pokemon]:
    """Build a team by filling 6 roles from curated pools with type diversity."""
    type_counts: dict[str, int] = {}
    team: list[Pokemon] = []
    used_ids: set[int] = set()

    roles = [
        (STATUS_LEAD_IDS, "status"),
        (PHYS_SWEEPER_IDS, "sweeper"),
        (SPEC_SWEEPER_IDS, "sweeper"),
        (PHYS_WALL_IDS, "wall"),
        (SPEC_WALL_IDS, "wall"),
    ]

    for pool_ids, role in roles:
        candidates = [sid for sid in pool_ids if sid not in used_ids
                      and (allowed_tiers is None or get_tier(sid) in allowed_tiers)]
        rng.shuffle(candidates)
        picked = False
        for sid in candidates:
            types = data.pokemon[sid]["types"]
            if not _type_ok(type_counts, types):
                continue
            if role == "status":
                status_mid = _pick_status_move(data, sid)
                if status_mid is None:
                    continue
                moves = [status_mid] + _pick_coverage_moves_with_exclude(data, sid, 3, {status_mid})
            elif role == "sweeper":
                # 50% chance: setup move + 3 coverage, else 4 coverage
                setup_mid = _pick_setup_move(data, sid)
                if setup_mid and rng.random() < 0.5:
                    moves = [setup_mid] + _pick_coverage_moves(data, sid, 3)
                else:
                    moves = _pick_coverage_moves(data, sid, 4)
            else:
                moves = _pick_wall_moves(data, sid)
            team.append(_make_pokemon(data, sid, moves))
            used_ids.add(sid)
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            picked = True
            break
        if not picked:
            return build_random_team(data, rng=rng, allowed_tiers=allowed_tiers)

    # ---- Wildcard ----
    all_pool = set(
        STATUS_LEAD_IDS + PHYS_SWEEPER_IDS + SPEC_SWEEPER_IDS
        + PHYS_WALL_IDS + SPEC_WALL_IDS
    )
    candidates = [sid for sid in all_pool if sid not in used_ids
                  and (allowed_tiers is None or get_tier(sid) in allowed_tiers)]
    rng.shuffle(candidates)
    for sid in candidates:
        types = data.pokemon[sid]["types"]
        if _type_ok(type_counts, types):
            moves = _pick_best_moves(data, sid, 4)
            team.append(_make_pokemon(data, sid, moves))
            return team
    return build_random_team(data, rng=rng, allowed_tiers=allowed_tiers)


def _build_ou_template(
    data: DataStore, rng: _random.Random, allowed_tiers: set[str] | None = None,
) -> list[Pokemon]:
    """Build from a template, filtered to allowed tiers."""
    if allowed_tiers is None:
        allowed_tiers = {"ou"}
    legal = [
        t for t in TEMPLATES
        if all(get_tier(sid) in allowed_tiers for sid, _ in t)
    ]
    if not legal:
        return _build_role_based(data, rng, allowed_tiers=allowed_tiers)
    template = rng.choice(legal)
    return [_make_pokemon(data, sid, mids) for sid, mids in template]


def _build_offensive_template(data: DataStore, rng: _random.Random) -> list[Pokemon]:
    """Build from an offensive template (4 damaging moves per mon)."""
    template = rng.choice(OFFENSIVE_TEMPLATES)
    return [_make_pokemon(data, sid, mids) for sid, mids in template]


def _pick_coverage_moves_with_exclude(
    data: DataStore, species_id: int, count: int, exclude: set[int],
) -> list[int]:
    """Pick coverage moves excluding specific move IDs."""
    learnset = set(data.pokemon[species_id]["learnset"])
    has_sleep = bool(learnset & _SLEEP_MOVES)
    skip = SELF_KO_MOVES | _MULTI_TURN_PENALTY | _JUNK_MOVES | exclude
    if not has_sleep:
        skip = skip | {_DREAM_EATER}
    damaging = [m for m in data.get_damaging_moves(species_id) if m not in skip]
    if not damaging:
        # fall back to including junk if nothing else available
        damaging = [m for m in data.get_damaging_moves(species_id) if m not in exclude]
    if not damaging:
        remaining = [m for m in learnset if m not in skip]
        return remaining[:count]
    damaging.sort(key=lambda m: _effective_power(data, species_id, m), reverse=True)
    return damaging[:count]


def _pick_coverage_moves(data: DataStore, species_id: int, count: int = 4,
                         exclude: set[int] | None = None,
                         has_sleep_override: bool | None = None) -> list[int]:
    """Pick damaging moves prioritizing type coverage, then effective power."""
    if exclude is None:
        exclude = set()
    learnset = set(data.pokemon[species_id]["learnset"])
    has_sleep = has_sleep_override if has_sleep_override is not None else bool(learnset & _SLEEP_MOVES)
    skip = SELF_KO_MOVES | _MULTI_TURN_PENALTY | _JUNK_MOVES | exclude
    if not has_sleep:
        skip = skip | {_DREAM_EATER}
    damaging = [m for m in data.get_damaging_moves(species_id) if m not in skip]
    if not damaging:
        # fall back to including junk
        damaging = [m for m in data.get_damaging_moves(species_id)
                    if m not in (SELF_KO_MOVES | _MULTI_TURN_PENALTY | exclude)]
    if not damaging:
        return list(data.pokemon[species_id]["learnset"])[:count]

    def eff_pow(mid):
        return _effective_power(data, species_id, mid)

    # group by type, pick the strongest per type (by effective power)
    by_type: dict[str, list[int]] = {}
    for mid in damaging:
        mdata = data.moves[mid]
        mtype = mdata.get("type", "normal")
        by_type.setdefault(mtype, []).append(mid)

    for t in by_type:
        by_type[t].sort(key=eff_pow, reverse=True)

    # greedily pick best move per type, prioritizing non-normal coverage
    chosen: list[int] = []
    used_types: set[str] = set()
    # first pass: non-normal types first for coverage diversity
    candidates = [(by_type[t][0], t) for t in by_type if by_type[t] and t != "normal"]
    candidates.sort(key=lambda x: eff_pow(x[0]), reverse=True)
    for mid, mtype in candidates:
        if len(chosen) >= count:
            break
        if eff_pow(mid) >= 30:
            chosen.append(mid)
            used_types.add(mtype)

    # second pass: fill with best remaining moves, preferring unused types
    if len(chosen) < count:
        remaining = [(mid, eff_pow(mid), data.moves[mid].get("type", "normal"))
                     for mid in damaging if mid not in chosen
                     and eff_pow(mid) >= 30]
        remaining.sort(key=lambda x: x[1], reverse=True)
        # prefer moves of types not yet chosen
        for mid, _, mtype in remaining:
            if len(chosen) >= count:
                break
            if mtype not in used_types:
                chosen.append(mid)
                used_types.add(mtype)
        # then fill any remaining with best power (allow duplicate types)
        for mid, _, mtype in remaining:
            if len(chosen) >= count:
                break
            if mid not in chosen:
                chosen.append(mid)

    return chosen[:count]


def build_offensive_team(
    data: DataStore,
    rng: _random.Random | None = None,
    allowed_tiers: set[str] | None = None,
) -> list[Pokemon]:
    """Build a team optimized for pure offense: 4 damaging moves with type coverage."""
    if rng is None:
        rng = _random.Random()

    # pick from sweeper pools -- mons with good offensive stats
    pool = list(set(PHYS_SWEEPER_IDS + SPEC_SWEEPER_IDS))
    if allowed_tiers:
        pool = [sid for sid in pool if get_tier(sid) in allowed_tiers]

    rng.shuffle(pool)
    type_counts: dict[str, int] = {}
    team: list[Pokemon] = []

    for sid in pool:
        if len(team) >= 6:
            break
        types = data.pokemon[sid]["types"]
        if not _type_ok(type_counts, types):
            continue
        moves = _pick_coverage_moves(data, sid, 4)
        team.append(_make_pokemon(data, sid, moves))
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

    # fill remaining slots from OU pool if needed
    if len(team) < 6:
        extra = [sid for sid in list(OU) if sid not in {p.species.id for p in team}
                 and (allowed_tiers is None or get_tier(sid) in allowed_tiers)]
        rng.shuffle(extra)
        for sid in extra:
            if len(team) >= 6:
                break
            types = data.pokemon[sid]["types"]
            if not _type_ok(type_counts, types):
                continue
            moves = _pick_coverage_moves(data, sid, 4)
            team.append(_make_pokemon(data, sid, moves))
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1

    return team


def build_team(
    data: DataStore,
    team_size: int = 6,
    rng: _random.Random | None = None,
    tier: str = "ou",
    strategy: str | None = None,
) -> list[Pokemon]:
    """Tier-constrained team builder: 50% template, 35% role-based, 15% random.

    tier controls which pokemon are allowed:
      "ou"   -- OU + UU + NU (standard, no ubers)
      "uu"   -- UU + NU only
      "uber" -- anything goes (legacy behavior)
      "all"  -- anything goes (same as uber)

    strategy overrides the build method:
      "offensive" -- 60% offensive template, 40% offensive random (4 damaging moves)
      None        -- default mixed builder
    """
    if rng is None:
        rng = _random.Random()

    if tier in ("uber", "all"):
        allowed = None  # no restriction
    elif tier == "uu":
        allowed = {"uu"}
    else:  # ou (default)
        allowed = {"ou"}

    # ---- Offensive strategy ----
    if strategy == "offensive":
        if rng.random() < 0.60:
            return _build_offensive_template(data, rng)
        return build_offensive_team(data, rng=rng, allowed_tiers=allowed)

    roll = rng.random()
    if roll < 0.50:
        if allowed is None:
            return _build_from_template(data, rng)
        return _build_ou_template(data, rng, allowed_tiers=allowed)
    elif roll < 0.85:
        return _build_role_based(data, rng, allowed_tiers=allowed)
    else:
        return build_random_team(data, team_size=team_size, rng=rng,
                                 allowed_tiers=allowed)
