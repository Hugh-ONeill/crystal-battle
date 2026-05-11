# Gen 9 OU sample teams from Smogon's official sample teams thread.
# source: https://www.smogon.com/forums/threads/sv-ou-sample-teams-new-samples-added-post-scl-and-olt.3712513/
# Each team is a different archetype to give the bench coverage across the meta.
# Mixed eras (pre/post Tera Blast ban, pre/post Gouging Fire ban) — bench targets
# eval behavior, not metagame legality.

SAMPLE_TEAMS_GEN9 = [
    # ---- 0: Sun Offense (Venusaur + Walking Wake) ----
    """Great Tusk @ Assault Vest
Ability: Protosynthesis
Tera Type: Water
EVs: 160 HP / 132 Atk / 12 SpD / 204 Spe
Adamant Nature
- Headlong Rush
- Rapid Spin
- Ice Spinner
- Close Combat

Walking Wake @ Wise Glasses
Ability: Protosynthesis
Tera Type: Ghost
EVs: 12 HP / 244 SpA / 252 Spe
Timid Nature
- Hydro Steam
- Draco Meteor
- Flamethrower
- Flip Turn

Kingambit @ Air Balloon
Ability: Supreme Overlord
Tera Type: Ghost
EVs: 140 HP / 252 Atk / 112 Spe
Adamant Nature
- Swords Dance
- Kowtow Cleave
- Iron Head
- Sucker Punch

Venusaur @ Life Orb
Ability: Chlorophyll
Tera Type: Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Growth
- Giga Drain
- Weather Ball
- Sludge Bomb

Ninetales @ Heat Rock
Ability: Drought
Tera Type: Ghost
EVs: 248 HP / 44 Def / 216 Spe
Timid Nature
IVs: 0 Atk
- Will-O-Wisp
- Weather Ball
- Healing Wish
- Encore

Raging Bolt @ Air Balloon
Ability: Protosynthesis
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
IVs: 20 Atk
- Thunderbolt
- Thunderclap
- Dragon Pulse
- Calm Mind""",

    # ---- 1: Bulky Offense (Garganacl + Hatterene) ----
    """Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Water
EVs: 252 HP / 16 Def / 216 SpD / 24 Spe
Careful Nature
- Salt Cure
- Stealth Rock
- Protect
- Recover

Darkrai @ Choice Specs
Ability: Bad Dreams
Tera Type: Poison
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Dark Pulse
- Psychic
- Focus Blast
- Sludge Bomb

Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Ice
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Headlong Rush
- Close Combat
- Ice Spinner
- Rapid Spin

Hatterene @ Rocky Helmet
Ability: Magic Bounce
Tera Type: Flying
EVs: 240 HP / 252 Def / 16 SpA
Relaxed Nature
IVs: 0 Spe
- Dazzling Gleam
- Future Sight
- Nuzzle
- Pain Split

Tornadus-Therian @ Assault Vest
Ability: Regenerator
Tera Type: Fairy
EVs: 184 HP / 16 Def / 68 SpD / 240 Spe
Timid Nature
- Bleakwind Storm
- Knock Off
- U-turn
- Heat Wave

Dragonite @ Silk Scarf
Ability: Multiscale
Tera Type: Normal
EVs: 152 HP / 252 Atk / 104 Spe
Adamant Nature
- Dragon Dance
- Roost
- Extreme Speed
- Earthquake""",

    # ---- 2: Balance (Gholdengo + Ting-Lu hazard stack) ----
    """Gholdengo @ Choice Scarf
Ability: Good as Gold
Tera Type: Steel
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Shadow Ball
- Make It Rain
- Thunder Wave
- Trick

Darkrai @ Heavy-Duty Boots
Ability: Bad Dreams
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Dark Pulse
- Ice Beam
- Knock Off
- Thunder Wave

Clefable @ Life Orb
Ability: Magic Guard
Tera Type: Water
EVs: 252 HP / 80 SpA / 176 Spe
Modest Nature
IVs: 0 Atk
- Calm Mind
- Moonblast
- Flamethrower
- Moonlight

Ting-Lu @ Red Card
Ability: Vessel of Ruin
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Stealth Rock
- Spikes
- Ruination
- Whirlwind

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Dance
- Extreme Speed
- Earthquake
- Roost

Pecharunt @ Heavy-Duty Boots
Ability: Poison Puppeteer
Tera Type: Ghost
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Nasty Plot
- Shadow Ball
- Malignant Chain
- Recover""",

    # ---- 3: Stall (Dondozo + Blissey + Toxapex) ----
    """Dondozo @ Leftovers
Ability: Unaware
Tera Type: Dragon
EVs: 248 HP / 252 Def / 8 Spe
Impish Nature
- Body Press
- Avalanche
- Rest
- Sleep Talk

Blissey @ Leftovers
Ability: Natural Cure
Tera Type: Dark
EVs: 20 HP / 252 Def / 236 SpD
Calm Nature
IVs: 0 Atk
- Seismic Toss
- Soft-Boiled
- Calm Mind
- Stealth Rock

Corviknight @ Leftovers
Ability: Pressure
Tera Type: Fighting
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
IVs: 0 Atk
- Iron Defense
- Defog
- Roost
- Body Press

Weezing-Galar @ Heavy-Duty Boots
Ability: Neutralizing Gas
Tera Type: Ghost
EVs: 252 HP / 244 Def / 12 Spe
Bold Nature
IVs: 0 Atk
- Will-O-Wisp
- Defog
- Pain Split
- Toxic Spikes

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 12 Def / 252 SpD
Careful Nature
IVs: 24 Spe
- Earthquake
- Spikes
- Knock Off
- Protect

Toxapex @ Leftovers
Ability: Regenerator
Tera Type: Steel
EVs: 252 HP / 252 SpD / 4 Spe
Calm Nature
- Toxic
- Recover
- Poison Jab
- Haze""",

    # ---- 4: Rain (Pelipper + Barraskewda + Archaludon) ----
    """Pelipper @ Damp Rock
Ability: Drizzle
Tera Type: Water
EVs: 248 HP / 48 Def / 212 SpD
Relaxed Nature
IVs: 0 Spe
- Surf
- U-turn
- Hurricane
- Roost

Barraskewda @ Choice Band
Ability: Swift Swim
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Liquidation
- Flip Turn
- Aqua Jet
- Close Combat

Kingambit @ Air Balloon
Ability: Supreme Overlord
Tera Type: Fairy
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Swords Dance
- Sucker Punch
- Kowtow Cleave
- Tera Blast

Archaludon @ Assault Vest
Ability: Stamina
Tera Type: Fairy
EVs: 252 HP / 176 SpA / 8 SpD / 72 Spe
Modest Nature
IVs: 0 Atk
- Draco Meteor
- Flash Cannon
- Electro Shot
- Body Press

Iron Treads @ Eject Button
Ability: Quark Drive
Tera Type: Ghost
EVs: 120 HP / 252 SpA / 136 Spe
Timid Nature
- Earth Power
- Rapid Spin
- Stealth Rock
- Volt Switch

Raging Bolt @ Booster Energy
Ability: Protosynthesis
Tera Type: Fairy
EVs: 196 HP / 252 SpA / 60 Spe
Modest Nature
IVs: 20 Atk
- Calm Mind
- Thunderclap
- Weather Ball
- Dragon Pulse""",

    # ---- 5: Trick Room (Hoopa-U + Ursaluna + dual setters) ----
    """Hoopa-Unbound @ Eject Pack
Ability: Magician
Tera Type: Dark
EVs: 248 HP / 252 Atk / 8 SpA
Brave Nature
IVs: 0 Spe
- Hyperspace Fury
- Drain Punch
- Psychic
- Trick Room

Ursaluna @ Flame Orb
Ability: Guts
Tera Type: Normal
EVs: 252 HP / 252 Atk / 4 SpD
Brave Nature
IVs: 0 Spe
- Facade
- Headlong Rush
- Fire Punch
- Swords Dance

Hatterene @ Focus Sash
Ability: Magic Bounce
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 SpA
Quiet Nature
IVs: 0 Atk / 0 Spe
- Dazzling Gleam
- Psychic
- Trick Room
- Healing Wish

Cresselia @ Leftovers
Ability: Levitate
Tera Type: Poison
EVs: 252 HP / 176 Def / 80 SpD
Calm Nature
IVs: 0 Atk / 0 Spe
- Moonblast
- Moonlight
- Trick Room
- Lunar Dance

Iron Hands @ Booster Energy
Ability: Quark Drive
Tera Type: Flying
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Wild Charge
- Drain Punch
- Ice Punch
- Swords Dance

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Flying
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Iron Head
- Sucker Punch
- Swords Dance""",

    # ---- 6: Screens Hyper Offense (Deoxys-Speed + Kingambit) ----
    """Deoxys-Speed @ Light Clay
Ability: Pressure
Tera Type: Dark
EVs: 248 HP / 8 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Taunt
- Light Screen
- Reflect
- Psycho Boost

Enamorus @ Choice Scarf
Ability: Contrary
Tera Type: Stellar
EVs: 4 Def / 252 SpA / 252 Spe
Modest Nature
IVs: 0 Atk
- Moonblast
- Earth Power
- Healing Wish
- Mystical Fire

Latias @ Leftovers
Ability: Levitate
Tera Type: Poison
EVs: 248 HP / 244 Def / 16 Spe
Timid Nature
IVs: 0 Atk
- Calm Mind
- Agility
- Stored Power
- Aura Sphere

Gouging Fire @ Booster Energy
Ability: Protosynthesis
Tera Type: Fairy
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Outrage
- Flare Blitz
- Morning Sun

Hatterene @ Leftovers
Ability: Magic Bounce
Tera Type: Steel
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
- Nuzzle
- Calm Mind
- Draining Kiss
- Stored Power

Kingambit @ Air Balloon
Ability: Supreme Overlord
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Kowtow Cleave
- Iron Head
- Sucker Punch""",

    # ---- 7: Sand (Tyranitar + Excadrill core) ----
    """Tyranitar @ Smooth Rock
Ability: Sand Stream
Tera Type: Flying
EVs: 248 HP / 16 Def / 24 SpA / 216 SpD / 4 Spe
Sassy Nature
- Knock Off
- Ice Beam
- Thunder Wave
- Stealth Rock

Hydrapple @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Poison
EVs: 208 HP / 172 Def / 88 SpA / 40 Spe
Bold Nature
IVs: 0 Atk
- Nasty Plot
- Giga Drain
- Fickle Beam
- Earth Power

Moltres @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Fairy
EVs: 248 HP / 248 Def / 12 Spe
Bold Nature
- Flamethrower
- U-turn
- Roost
- Roar

Zamazenta @ Leftovers
Ability: Dauntless Shield
Tera Type: Fire
EVs: 4 Atk / 252 Def / 252 Spe
Jolly Nature
- Body Press
- Crunch
- Iron Defense
- Roar

Slowking-Galar @ Assault Vest
Ability: Regenerator
Tera Type: Grass
EVs: 248 HP / 184 Def / 28 SpA / 48 Spe
Bold Nature
IVs: 0 Atk
- Psyshock
- Sludge Bomb
- Flamethrower
- Ice Beam

Excadrill @ Air Balloon
Ability: Sand Rush
Tera Type: Fire
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Earthquake
- Iron Head
- Rapid Spin
- Swords Dance""",

    # ---- 8: Sticky Web HO (Ribombee + Manaphy + Roaring Moon) ----
    """Ribombee @ Focus Sash
Ability: Shield Dust
Tera Type: Ghost
EVs: 8 HP / 136 Def / 196 SpA / 168 Spe
Timid Nature
IVs: 0 Atk
- Sticky Web
- Moonblast
- Stun Spore
- Psychic Noise

Glimmora @ Power Herb
Ability: Toxic Debris
Tera Type: Fairy
EVs: 16 Def / 240 SpA / 252 Spe
Modest Nature
IVs: 0 Atk
- Sludge Bomb
- Meteor Beam
- Earth Power
- Dazzling Gleam

Manaphy @ Leftovers
Ability: Hydration
Tera Type: Fairy
EVs: 112 HP / 216 SpA / 180 Spe
Modest Nature
IVs: 0 Atk
- Surf
- Stored Power
- Acid Armor
- Tail Glow

Zamazenta @ Mirror Herb
Ability: Dauntless Shield
Tera Type: Fire
EVs: 8 HP / 60 Atk / 252 Def / 188 Spe
Jolly Nature
- Body Press
- Iron Head
- Stone Edge
- Iron Defense

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fairy
EVs: 248 HP / 16 Def / 72 SpA / 172 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Hex
- Recover
- Thunder Wave

Roaring Moon @ Booster Energy
Ability: Protosynthesis
Tera Type: Flying
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Knock Off
- Earthquake
- Acrobatics
- Dragon Dance""",

    # ---- 9: Kingambit Hyper Offense (Cresselia + SD wincon) ----
    """Kingambit @ Lum Berry
Ability: Supreme Overlord
Tera Type: Fairy
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Kowtow Cleave
- Sucker Punch
- Iron Head

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Dark
EVs: 24 Atk / 232 SpA / 252 Spe
Naive Nature
- Moonblast
- Close Combat
- Destiny Bond
- Encore

Cresselia @ Leftovers
Ability: Levitate
Tera Type: Poison
EVs: 252 HP / 172 Def / 64 Spe
Bold Nature
IVs: 0 Atk
- Calm Mind
- Stored Power
- Moonblast
- Moonlight

Samurott-Hisui @ Focus Sash
Ability: Sharpness
Tera Type: Water
EVs: 232 Atk / 24 Def / 252 Spe
Jolly Nature
- Ceaseless Edge
- Razor Shell
- Aqua Jet
- Sacred Sword

Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Ice
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Bulk Up
- Ice Spinner
- Headlong Rush
- Close Combat

Sneasler @ Air Balloon
Ability: Unburden
Tera Type: Flying
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Close Combat
- Acrobatics
- Night Slash""",
]

# Indices into SAMPLE_TEAMS_GEN9 that are "specialty" archetypes — they distort
# round-robin standings because they produce mostly draws against the engine
# at typical search budgets. Exclude from default round-robin standings;
# include only when explicitly testing wallbreaking.
#   3 (Stall): 82% draw rate vs the rest of the pool at 200ms / 120 max-turns.
SPECIALTY_TEAM_INDICES = {3}


# Gen 9 Ubers sample teams — used as "neutrally strong opponents" yardstick.
# Source: https://www.smogon.com/forums/threads/sv-ubers-sample-teams.3736089/
# These contain restricted legendaries and are not legal in OU. Use them as a
# fixed strong-opponent gauntlet to compare how OU teams hold up.
UBERS_TEAMS_GEN9 = [
    # ---- 0: Specs Miraidon Balance (Miraidon + Giratina-O + Ho-Oh) ----
    """Miraidon @ Choice Specs
Ability: Hadron Engine
Tera Type: Fairy
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- U-turn
- Electro Drift
- Draco Meteor
- Overheat

Ho-Oh @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Fairy
EVs: 248 HP / 240 Def / 20 Spe
Impish Nature
- Sacred Fire
- Brave Bird
- Whirlwind
- Recover

Giratina-Origin @ Griseous Core
Ability: Levitate
Tera Type: Steel
EVs: 248 HP / 200 Atk / 60 Spe
Adamant Nature
- Poltergeist
- Defog
- Shadow Sneak
- Dragon Tail

Necrozma-Dusk-Mane @ Heavy-Duty Boots
Ability: Prism Armor
Tera Type: Dark
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Sunsteel Strike
- Knock Off
- Stealth Rock
- Moonlight

Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Steel
EVs: 248 HP / 252 SpD / 4 Spe
Careful Nature
- Earthquake
- Ruination
- Spikes
- Whirlwind

Koraidon @ Choice Scarf
Ability: Orichalcum Pulse
Tera Type: Fire
EVs: 72 HP / 252 Atk / 4 Def / 180 Spe
Adamant Nature
- Low Kick
- Dragon Claw
- U-turn
- Flare Blitz""",

    # ---- 1: Kyogre+Groudon Balance (weather + Zacian + Arceus core) ----
    """Koraidon @ Heavy-Duty Boots
Ability: Orichalcum Pulse
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Flame Charge
- Low Kick
- Outrage

Arceus @ Heavy-Duty Boots
Ability: Multitype
Tera Type: Fire
EVs: 184 HP / 252 Atk / 72 Spe
Adamant Nature
- Swords Dance
- Extreme Speed
- Earthquake
- Recover

Ho-Oh @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Grass
EVs: 248 HP / 252 Def / 8 Spe
Impish Nature
- Sacred Fire
- Brave Bird
- Whirlwind
- Recover

Zacian-Crowned @ Rusted Sword
Ability: Intrepid Sword
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Behemoth Blade
- Wild Charge
- Crunch

Groudon @ Leftovers
Ability: Drought
Tera Type: Ghost
EVs: 248 HP / 244 Def / 16 Spe
Impish Nature
- Spikes
- Stealth Rock
- Precipice Blades
- Will-O-Wisp

Kyogre @ Heavy-Duty Boots
Ability: Drizzle
Tera Type: Fairy
EVs: 248 HP / 164 Def / 56 SpA / 20 SpD / 20 Spe
Bold Nature
IVs: 0 Atk
- Thunder Wave
- Origin Pulse
- Ice Beam
- Thunder""",

    # ---- 2: Double Priority Hyper Offense (Deoxys-S lead + Espeed/Sucker) ----
    """Deoxys-Speed @ Mental Herb
Ability: Pressure
Tera Type: Ghost
EVs: 248 HP / 144 Def / 116 Spe
Timid Nature
IVs: 0 Atk
- Spikes
- Taunt
- Thunder Wave
- Skill Swap

Koraidon @ Loaded Dice
Ability: Orichalcum Pulse
Tera Type: Fire
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Swords Dance
- Flare Blitz
- Scale Shot
- Taunt

Zacian-Crowned @ Rusted Sword
Ability: Intrepid Sword
Tera Type: Fighting
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Swords Dance
- Behemoth Blade
- Close Combat
- Wild Charge

Arceus @ Life Orb
Ability: Multitype
Tera Type: Normal
EVs: 68 HP / 252 Atk / 188 Spe
Adamant Nature
- Swords Dance
- Extreme Speed
- Shadow Claw
- Double-Edge

Lunala @ Power Herb
Ability: Shadow Shield
Tera Type: Ghost
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Agility
- Moongeist Beam
- Psyshock
- Meteor Beam

Kingambit @ Black Glasses
Ability: Supreme Overlord
Tera Type: Fire
EVs: 240 HP / 252 Atk / 16 Spe
Adamant Nature
- Swords Dance
- Sucker Punch
- Kowtow Cleave
- Iron Head""",
]

