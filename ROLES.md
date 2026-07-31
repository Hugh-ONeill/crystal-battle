# Role annotations — review copy

Generated from `showdown/roles.json` (34 entries). **Edit the JSON, not this file.**

Nothing consumes these yet. **The plain paragraph under each entry is the `fact`** — the only field a consumer would ever be shown. It is written to stand alone: present tense, mechanically true, no named opponents, no references to other entries. Everything about where a claim came from lives in the collapsed provenance block, which no consumer sees.

| grade | meaning |
|---|---|
| measured here | backed by a measurement from this campaign |
| Smogon-cited | grounded in retrieved Smogon analysis |
| USER-CORRECTED | you corrected my reading — highest confidence |
| UNVERIFIED | my inference only — scrutinise these |

## At a glance

| usage | mon | grade | preserve | deployment | tags |
|---:|---|---|---|---|---|
| 32.9% | **greattusk** ⚡ | Smogon-cited | med | — | hazard-removal, wall, pivot |
| 22.5% | **kingambit** | measured here | high | late-cleaner | cleaner, wincon |
| 21.9% | **gholdengo** ⚡ ✳ | USER-CORRECTED | high | None | spinblocker, glue |
| 17.9% | **dragonite** | Smogon-cited | high | setup-window | wincon, setup-sweeper |
| 15.3% | **ironvaliant** ✳ | Smogon-cited | med | — | wallbreaker, setup-sweeper |
| 15.3% | **ragingbolt** | Smogon-cited | med | late-cleaner | wallbreaker, priority-attacker |
| 15.2% | **zamazenta** | Smogon-cited | high | — | wall, wincon, glue |
| 15.1% | **ogerponwellspring** | Smogon-cited | med | — | wallbreaker, sweeper |
| 14.6% | **dragapult** | Smogon-cited | med | pivot-cycle | pivot, wallbreaker |
| 12.9% | **hatterene** ⚡ | USER-CORRECTED | med | bait-switch | wall, sacrificial-support |
| 12.8% | **corviknight** ⚡ | Smogon-cited | med | pivot-cycle | hazard-removal, pivot, wall |
| 12.5% | **slowkinggalar** | measured here | med | pivot-cycle | pivot, wall |
| 11.2% | **gliscor** | measured here | high | pivot-cycle | annuity, wall |
| 11.0% | **irontreads** | Smogon-cited | med | lead | hazard-removal, hazard-setter, lead |
| 10.6% | **kyurem** ✳ | Smogon-cited | med | — | wallbreaker |
| 10.3% | **samurotthisui** ⚡ | Smogon-cited | med | lead | hazard-setter, wallbreaker |
| 10.0% | **cinderace** | Smogon-cited | med | — | pivot, hazard-control |
| 9.6% | **tinglu** | measured here | med | lead | hazard-setter, wall |
| 9.3% | **landorustherian** | Smogon-cited | high | pivot-cycle | pivot, wall, glue |
| 9.2% | **ceruledge** ▸ | USER-CORRECTED | med | setup-window | setup-sweeper, wincon |
| 9.2% | **pecharunt** | measured here | med | — | wall, status-spreader |
| 9.0% | **glimmora** | Smogon-cited | low | lead | suicide-lead, hazard-setter |
| 8.9% | **ironmoth** | Smogon-cited | med | — | setup-sweeper |
| 8.9% | **alomomola** | Smogon-cited | high | pivot-cycle | pivot, annuity, wall |
| 8.1% | **darkrai** | Smogon-cited | med | late-cleaner | revenge-killer, wallbreaker |
| 7.3% | **walkingwake** | measured here | med | — | weather-abuser, sweeper |
| 7.3% | **rillaboom** | Smogon-cited | med | — | terrain-setter |
| 7.0% | **zapdos** | Smogon-cited | med | pivot-cycle | pivot |
| 6.3% | **deoxysspeed** | Smogon-cited | low | lead | suicide-lead, screens-setter, hazard-setter |
| 6.2% | **garganacl** | Smogon-cited | high | setup-window | wall, wincon |
| 4.7% | **pelipper** | measured here | high | lead | weather-setter |
| 3.3% | **torkoal** | measured here | med | lead | weather-setter |
| 2.6% | **barraskewda** | measured here | med | — | weather-abuser, sweeper |
| 1.8% | **grimmsnarl** | Smogon-cited | low | lead | suicide-lead, screens-setter |

⚡ = value is conditional  ✳ = splits into distinct sets  ▸ = has a written play sequence

## Entries

### greattusk — 32.9% usage · *Smogon-cited*

**tags** hazard-removal, wall, pivot · **ability** protosynthesis · **preserve** med

Rapid Spin removes entry hazards for the whole team. Protosynthesis raises its Speed, letting it outpace threats that would normally be faster. Its removal is a service to teammates, not a personal benefit.

**Conditional — `opponent_has:hazard-setter`:** preserve → high

> Hazard removal is only worth something while the opponent is setting hazards. Against a team that sets none, the removal move is a wasted slot.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Great Tusk (gen9ou) — Offensive Utility / Defensive]: "Rapid Spin allows it to remove entry hazards for its teammates, and the Speed boost lets it scare out usually faster threats like Darkrai and Iron Crown"; also "a staple on Sticky Web teams". SOLE-REMOVER CAVEAT: preserve:high holds only when it is the team's only remover — a TEAM property this per-species file cannot express (schema gap). Hazard chip is our pinned loss mechanism (SR = 21% of damage taken), so the remover is a team-persistent resource. [SCHEMA GAP, reviewer note: preserve also depends on whether OUR OWN team carries another remover — a team property this per-species file cannot express, and `conditional` only covers opponent facts. Moved out of the consumer-facing note 2026-07-31.]

</details>

### kingambit — 22.5% usage · *measured here*

**tags** cleaner, wincon · **ability** supremeoverlord · **preserve** high · **deployment** late-cleaner · **lead_intent** avoid · **value_curve** grows_with_own_faints

Supreme Overlord raises its power by 10% for each fallen ally, so it is weakest at full team and strongest last. Sucker Punch gives it priority. Spending it early forfeits the scaling it exists for.

<details><summary>provenance (review only — not shown to any consumer)</summary>

Supreme Overlord = +10% power per fallen ally (abilities.rs:2257), now mirrored into threat_vs (c68f15a). Timing instrument: we first deploy it ~T9 with 0.44 allies down and 77% of the time with ZERO down, vs the ladder population's T13.0 / 0.96. fp is equally guilty, so this is absolute strength, not fp-gap.

</details>

### gholdengo — 21.9% usage · *USER-CORRECTED*

**tags** spinblocker, glue · **ability** goodasgold · **preserve** high

Good as Gold makes it immune to status moves. Its Ghost typing blocks Rapid Spin, so hazards its side has set cannot be spun away while it is alive — this is true of every set, being a property of its typing rather than its moves. Air Balloon, its most common item, adds a temporary Ground immunity.

**Set — Nasty Plot:** **tags** setup-sweeper, wallbreaker · **preserve** high · **deployment** setup-window

> Nasty Plot doubles its Special Attack, then Shadow Ball and Make It Rain sweep. Recover sustains it across the game.

**Set — Choice Scarf:** **tags** revenge-killer · **preserve** med · **deployment** late-cleaner

> A Choice Scarf makes it fast enough to revenge-kill, and Trick can pass that Scarf to a wall to cripple it.

**Set — Status Hex:** **tags** status-spreader, wall · **preserve** med · **deployment** pivot-cycle

> Thunder Wave or Will-O-Wisp inflicts status, then Hex doubles in power against the statused target.

**Conditional — `board:hazards_up AND opponent_has:hazard-removal`:** 

> The spinblock only cashes in when hazards are actually standing that the other side wants to clear. The typing blocks Rapid Spin regardless, but with an empty field or an opponent carrying no removal it changes nothing, and the mon is being kept for its attacking sets instead.

<details><summary>provenance (review only — not shown to any consumer)</summary>

USER-CORRECTED 2026-07-31: my entry reduced a top-3 usage Pokemon to 'spinblocker', omitting that it is a primary win condition. Usage data confirms the correction — Shadow Ball 21.1% and Make It Rain 20.8% are its two most-used moves, Nasty Plot 12.8%, Recover 12.5%, Trick 8.4% (Choice Scarf is 33% of its items), Thunder Wave 4.1% with Hex 3.1%, Air Balloon 44% of items. Smogon's analysis calls it "both a defensive and offensive glue on virtually any team" and carries a dedicated Offensive Nasty Plot set. SCHEMA LIMITATION this exposed: the file annotates a SPECIES, but this mon has several sets with genuinely different roles (Nasty Plot sweeper, Choice Scarf Trick, defensive status/Hex). Tags and fact now cover all of them, which is lossy — a consumer cannot tell WHICH set is in front of it. The set-inference tiers already identify that at runtime, so a future consumer should key role off the inferred set rather than the species alone. Same class of gap as the sole-remover team-property problem. [Refined 2026-07-31 (user): spinblocking is a TYPING property so the tag is unconditional — what is conditional is whether it matters, which needs hazards actually on the field, not merely a hazard-setter on the roster. Split into per-set entries at the same time.]

</details>

### dragonite — 17.9% usage · *Smogon-cited*

**tags** wincon, setup-sweeper · **ability** multiscale · **preserve** high · **deployment** setup-window · **entry_condition** full_hp · **value_curve** decays_with_chip

Multiscale halves incoming damage only at full HP, so it needs an undamaged entry to function. Roost restores that condition. Dragon Dance turns one safe turn into a sweep.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Dragonite (gen9ou)]: Roost-based sets "give Dragonite easy setup opportunities" and let it check Ogerpon-W, Iron Moth and Rillaboom, "with Multiscale intact". Same entry_condition family as Ceruledge: Multiscale only halves damage at FULL HP, so entry chip deletes the defensive half of the wincon before it sets up — which is why the standard set runs Boots. The eval prices Multiscale when at full HP but cannot price PRESERVING full HP for a later sweep.

</details>

### ironvaliant — 15.3% usage · *Smogon-cited*

**tags** wallbreaker, setup-sweeper · **ability** quarkdrive · **preserve** med

High mixed offense with Booster Energy spent on entry to raise its best stat. Choice Specs Moonblast breaks special walls that normally check it.

**Set — Choice Specs:** **tags** wallbreaker · **preserve** med

> Choice Specs Moonblast breaks special walls that would otherwise check it.

**Set — Swords Dance:** **tags** setup-sweeper · **preserve** high · **deployment** setup-window

> Swords Dance doubles its Attack for a physical sweep, with Shadow Sneak as priority coverage.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Iron Valiant (gen9ou)]: "Choice Specs takes advantage of Moonblast's spammability and lets Iron Valiant break through and even 2HKO otherwise common Iron Valiant answers such as Assault Vest Hatterene, Galarian Weezing, and specially defensive Gliscor." Booster Energy is its top item by usage — a one-shot entry resource like Iron Moth's, so entry TIMING is the unpriced question.

</details>

### ragingbolt — 15.3% usage · *Smogon-cited*

**tags** wallbreaker, priority-attacker · **ability** protosynthesis · **preserve** med · **deployment** late-cleaner

Thunderclap is a priority move, so it revenge-kills faster threats and beats slower priority attackers. Its value is largely in being held back until something needs killing.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Raging Bolt (gen9ou) — Choice Specs]: "Thunderclap is used as powerful priority to revenge kill or force out faster threats such as Ogerpon-W, Iron Valiant, and Enamorus, and it also helps Raging Bolt beat slower priority users such as Kingambit and Scizor." Priority-in-reserve is a role the leaf eval sees only as a weak move until the turn it matters — same shape as Ceruledge's Shadow Sneak.

</details>

### zamazenta — 15.2% usage · *Smogon-cited*

**tags** wall, wincon, glue · **ability** dauntlessshield · **preserve** high

Dauntless Shield raises its Defense on entry. Bulk and typing let it check many physical attackers, so teams are often built expecting it to hold that job all game.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Zamazenta (gen9ou)]: "Zamazenta is the tier's most consistent glue piece and wincon. Its natural bulk alongside a valuable typing, which can be boosted further by Dauntless Shield, lets it take on many manner of offensive threats like Kingambit, Hisuian Samurott, some Dragonite sets, and Great Tusk." Glue = it is what other members are built around; losing it costs more than its own HP.

</details>

### ogerponwellspring — 15.1% usage · *Smogon-cited*

**tags** wallbreaker, sweeper · **ability** waterabsorb · **preserve** med

Ivy Cudgel and Power Whip give it high physical power, and Water Absorb heals it from Water moves. Trailblaze trades coverage for a Speed boost to sweep.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Ogerpon-Wellspring (gen9ou)]: "one of OU's premier physical attackers" with "the high power of Ivy Cudgel and Power Whip, the option to trade the latter with Trailblaze to outspeed otherwise faster foes". Trailblaze makes it a conditional sweeper — a setup curve the eval prices only one turn at a time.

</details>

### dragapult — 14.6% usage · *Smogon-cited*

**tags** pivot, wallbreaker · **ability** infiltrator · **preserve** med · **deployment** pivot-cycle

Infiltrator ignores Substitutes and screens. Very high Speed with U-turn makes it a momentum pivot rather than a mon to hold back.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Dragapult (gen9ou) — Boots Pivot]: "Infiltrator is the preferred ability, as it allows Dragapult to both hit and cripple foes regardless of" substitutes and screens. Infiltrator ignoring Substitute/screens is a matchup fact the eval does model; the role note is that it is the momentum piece, not a mon to hold back.

</details>

### hatterene — 12.9% usage · *USER-CORRECTED*

**tags** wall, sacrificial-support · **ability** magicbounce · **preserve** med · **deployment** bait-switch · **lead_intent** avoid

Magic Bounce reflects hazards and status back at the user, so it PREVENTS hazards rather than removing them — it must be in play when the setter acts. Healing Wish trades it away to fully restore a teammate.

**Conditional — `opponent_has:hazard-setter`:** preserve → high · tags → ['hazard-denial', 'wall', 'sacrificial-support']

> Magic Bounce only does something if the opponent actually carries hazard or status moves to reflect. Against a team that sets hazards it is the reason those hazards never land; against one that does not, it is an ordinary bulky attacker.

> It is switched in on a turn the setter is expected to act, so the hazard move is reflected back onto them. Inviting that attempt is the point, so only bringing it in when it is safe never collects the value.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Hatterene (gen9ou) — Healing Wish]: Magic Bounce denies hazards passively while alive; "Psychic Noise notably lets Hatterene significantly chip bulky foes like Gliscor, Garganacl, and Clefable, preventing them from recovering their health", and the Healing Wish set trades itself to "pivot them in safely". TWO team-persistent effects the eval cannot price: (a) Magic Bounce is hazard PREVENTION, worth most against exactly the hazard-stack that beats us — its death re-opens the SR war; (b) Healing Wish is a deliberate SACRIFICE that restores a teammate, so a low-HP Hatterene is not obviously a liability. Psychic Noise blocking recovery also matters in the long grinds where the recovery war is even. CONDITIONAL + DEPLOYMENT added 2026-07-31 (user): the static schema could not express that Magic Bounce's worth depends on the OPPONENT's roster, nor that the mon is played as a setup-baiting switch-in rather than a lead.

</details>

### corviknight — 12.8% usage · *Smogon-cited*

**tags** hazard-removal, pivot, wall · **ability** pressure · **preserve** med · **deployment** pivot-cycle

Defog clears hazards from both sides. Pressure makes opponents spend two PP per move. Roost and high Defense let it re-enter repeatedly.

**Conditional — `opponent_has:hazard-setter`:** preserve → high

> Hazard removal is only worth something while the opponent is setting hazards. Against a team that sets none, the removal move is a wasted slot.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Corviknight (gen9ou) — Defensive]: "Defog allows Corviknight to fulfill the role as the team's hazard remover", and Pressure "more quickly stall[s] out their attacks' PP" against setup sweepers (Dragonite, Gliscor, Kingambit). Sole-remover caveat still applies: preserve pressure is a TEAM property this per-species file cannot express. PP-stalling matters in the long grinds where the stall audit measured real PP bankruptcy. [SCHEMA GAP, reviewer note: preserve also depends on whether OUR OWN team carries another remover — a team property this per-species file cannot express, and `conditional` only covers opponent facts. Moved out of the consumer-facing note 2026-07-31.]

</details>

### slowkinggalar — 12.5% usage · *measured here*

**tags** pivot, wall · **ability** regenerator · **preserve** med · **deployment** pivot-cycle · **lead_intent** neutral

Regenerator restores a third of its HP whenever it switches out, so repeated entries are nearly free. Chilly Reception and Future Sight support pivoting. Its standard spread invests little in Speed.

<details><summary>provenance (review only — not shown to any consumer)</summary>

Regenerator pivots are net-positive on entry (fp routes hazard cycles through them: pays 12.5%, regains 33%) — the hazard_cycle finding. Also the mon whose speed floor produced the false Choice Scarf call (88f1f17), so its canonical spread is bulky with ~0 Speed.

</details>

### gliscor — 11.2% usage · *measured here*

**tags** annuity, wall · **ability** poisonheal · **preserve** high · **deployment** pivot-cycle · **value_curve** grows_with_own_status

Poison Heal turns being poisoned into recurring healing, so it must be statused before it works — activating the Toxic Orb is the enabling step, and losing the orb beforehand disables the whole set. Protect and Substitute stall while it heals.

<details><summary>provenance (review only — not shown to any consumer)</summary>

The flagship annuity case. Stall audit: fp's Poison Heal Gliscor played its FULL PP budget and generated 22.8 mons of free healing over three marathons while ours clicked 6 moves in ~711 turns. Eval terms shipped for exactly this (poke-engine 76af1e9: POISON_HEAL_STATUSED 15->35, PENDING +15) and our Gliscor now SubToxes at full budget. Known race: the naked orb walked into Knock Off on entry in 2/3 marathons, so activation timing is itself the play.

</details>

### irontreads — 11.0% usage · *Smogon-cited*

**tags** hazard-removal, hazard-setter, lead · **ability** quarkdrive · **preserve** med · **deployment** lead · **lead_intent** strong

Rapid Spin removes hazards and Stealth Rock sets them, so it can do either job. Quark Drive raises a stat on entry.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Iron Treads (gen9ou) — Lead]: The Lead set trades itself for hazard control and "gives it another way to block Defog and Rapid Spin"; coverage choice is explicitly about stopping Raging Bolt boosting or "Pecharunt from blocking Rapid Spin". Dual hazard role (sets AND removes) makes it a lead by design rather than a mon to preserve.

</details>

### kyurem — 10.6% usage · *Smogon-cited*

**tags** wallbreaker · **ability** pressure · **preserve** med

High mixed offense across several very different sets, so which set it is changes what counters it. Freeze-Dry hits Water types super effectively.

**Set — Choice Specs:** **tags** wallbreaker · **preserve** med

> Choice Specs locks it into one move but lets it break walls outright.

**Set — Loaded Dice Icicle Spear:** **tags** wallbreaker · **preserve** med

> A multi-hit physical set that breaks through Substitutes and Focus Sash.

**Set — Substitute Roost:** **tags** wall, setup-sweeper · **preserve** med · **deployment** setup-window

> Substitute and Roost let it wear down passive teams while avoiding status.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Kyurem (gen9ou)]: "one of SV OU's strongest offensive Pokemon, utilizing its powerful mixed offensive stats and solid overall bulk to run a wide variety of sets, each requiring different counterplay. Choice Specs makes Kyurem into a wallbreaker." 'Each set requires different counterplay' is a direct statement that set inference matters more than usual here — relevant to the belief-tier work. [Split into sets 2026-07-31: its own analysis says each set 'requires different counterplay', so a single species role is misleading here.]

</details>

### samurotthisui — 10.3% usage · *Smogon-cited*

**tags** hazard-setter, wallbreaker · **ability** sharpness · **preserve** med · **deployment** lead · **lead_intent** strong

Ceaseless Edge sets Spikes as part of an attack, so it lays hazards even through Magic Bounce and Taunt, which stop ordinary setters. Sharpness boosts its cutting moves.

**Conditional — `opponent_has:hazard-denial`:** preserve → high

> Ceaseless Edge sets Spikes as part of an attack, so it still lays hazards against opponents carrying Magic Bounce or Taunt, which stop moves like Stealth Rock and Spikes.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Samurott-Hisui (gen9ou)]: "one of the most prolific entry hazard setters in the tier with its signature move Ceaseless Edge, letting it both set Spikes and deal heavy damage thanks to its Sharpness ability. It can even do so in the face of Hatterene's Magic Bounce and Taunt." Setting hazards THROUGH Magic Bounce is the counter to hazard-denial — the other side of the war that decides our long games.

</details>

### cinderace — 10.0% usage · *Smogon-cited*

**tags** pivot, hazard-control · **ability** libero · **preserve** med

Court Change swaps all side conditions, moving hazards onto the opponent instead of clearing them. Libero changes its type to match its move; U-turn pivots.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Cinderace (gen9ou) — Offensive Pivot]: Recommended set is "Pyro Ball; U-turn; Court Change; ..." with Heavy-Duty Boots and Libero. COURT CHANGE is hazard control by THEFT — it flips our hazards onto them rather than clearing them, which is strictly better in a chip war and is why our ah3 team runs it. The eval sees a side-condition swap but not that it converts their investment into ours.

</details>

### tinglu — 9.6% usage · *measured here*

**tags** hazard-setter, wall · **ability** vesselofruin · **preserve** med · **deployment** lead · **lead_intent** strong

Vessel of Ruin weakens opposing Special Attack. It sets Stealth Rock and Spikes, and Whirlwind forces switches so those hazards keep applying.

<details><summary>provenance (review only — not shown to any consumer)</summary>

richwoman's most-used lead by a distance (45 of her games) and the anchor of the hazard-stack that produces our pinned chip loss. Her revealed set is SR/Whirlwind/Earthquake/Ruination — Whirlwind is what converts her hazards into forced-switch chip.

</details>

### landorustherian — 9.3% usage · *Smogon-cited*

**tags** pivot, wall, glue · **ability** intimidate · **preserve** high · **deployment** pivot-cycle

Intimidate lowers the opposing Attack every time it enters, so repeated switch-ins compound the effect. U-turn pivots; Earthquake and Stealth Rock add utility.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Landorus-Therian (gen9ou)]: "one of OU's premier pivots. It can check a wide variety of threats such as Kingambit, Ceruledge, and Raging Bolt." Intimidate is a persistent team-wide defensive effect applied on every entry — an annuity the eval prices only as a one-off stat drop.

</details>

### ceruledge — 9.2% usage · *USER-CORRECTED*

**tags** setup-sweeper, wincon · **ability** weakarmor · **preserve** med · **deployment** setup-window · **entry_condition** full_hp · **value_curve** decays_with_chip

Weak Armor raises its Speed when it is hit. Bitter Blade heals it for half the damage dealt. Focus Sash survives one hit but only from full HP. Shadow Sneak provides priority.

**The play:**

1. enter at FULL HP on a free switch — Focus Sash only functions from full, so entry chip cancels the plan
2. Swords Dance while they attack or switch
3. TAKE the hit: Sash holds at 1 HP, which procs Weak Armor for +Speed (losing the Sash here is the mechanism, not a loss)
4. now outspeeding: attack with Bitter Blade, whose drain heals back off 1 HP
5. hold Shadow Sneak in reserve for anything that still outspeeds or carries priority

> Every step scores badly on its own — 1 HP, no item, lowered Defense — and only the finished chain is a sweep. Judging any single turn of it in isolation rejects the plan.

<details><summary>provenance (review only — not shown to any consumer)</summary>

TWICE-CORRECTED, and the final read is the user's (2026-07-31). (1) My inference: 'Focus Sash is a resource chip destroys, so preserve it.' (2) RAG partly corrected it [smogon#Ceruledge (gen9ou) — Swords Dance]: popping the Sash with weak moves "would only give it a trouble-free Weak Armor boost", so the opponent removing the Sash is often GOOD for Ceruledge — I then over-corrected to preserve:low. (3) The real mechanism is a SEQUENCE, and entry chip is what breaks it: come in at FULL HP (Focus Sash only functions from full), Swords Dance, TAKE a hit, survive at 1 HP on Sash, which procs Weak Armor for +Speed, then outspeed and heal back with Bitter Blade's drain, holding Shadow Sneak in reserve for anything that still outspeeds or carries priority. So the Sash is not a thing to protect and not a thing to spend — it is the PIVOT of a multi-turn plan whose precondition is an uncontested entry. Any Stealth Rock chip kills the plan before turn one. DIRECTLY TIED to our pinned loss mechanism (SR = 21% of damage taken in the long grinds): this is a mon whose wincon the hazard war silently deletes, which is also why the standard set wants removal or Boots support. The whole sequence is exactly the beyond-horizon multi-turn plan the leaf eval cannot price. [Cross-references removed from the consumer-facing note 2026-07-31: it had pointed at Hatterene's bait-switch and the Curse/Dondozo accumulation. Those parallels are real and useful to a reviewer, but each entry must stand alone for a consumer.]

</details>

### pecharunt — 9.2% usage · *measured here*

**tags** wall, status-spreader · **ability** poisonpuppeteer · **preserve** med

Poison Puppeteer confuses any foe it poisons. Bulk and recovery let it stall and spread status.

<details><summary>provenance (review only — not shown to any consumer)</summary>

CORRECTION 2026-07-31 (user-flagged, verified): Poison Puppeteer IS implemented — generate_instructions.rs:907 adds CONFUSION to a foe it poisons, landed in 2e000e6 alongside Rock Head/Anger Shell/Synchronize. An earlier memory note listing it as a silent OU gap was stale. The engine models it correctly; this is now an ordinary role entry.

</details>

### glimmora — 9.0% usage · *Smogon-cited*

**tags** suicide-lead, hazard-setter · **preserve** low · **deployment** lead · **lead_intent** strong

Toxic Debris sets Toxic Spikes when it is hit by a physical move. Stealth Rock and Mortal Spin give it hazard control, and Focus Sash guarantees it gets at least one layer down.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Glimmora (gen9ou) — Lead]: the Lead set exists to deny setup — it stops "Ting-Lu from setting up entry hazards multiple times and Hatterene from trying to deny Stealth Rock with Magic Bounce". Confirms the suicide-lead role: its job is completed on the turns it survives, not by surviving. Directly relevant to richwoman, whose hazard stack is anchored by Ting-Lu (45 leads).

</details>

### ironmoth — 8.9% usage · *Smogon-cited*

**tags** setup-sweeper · **ability** quarkdrive · **preserve** med

Booster Energy is spent on entry to raise its best stat. Fiery Dance can raise its Special Attack further, letting it sweep once boosted.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Iron Moth (gen9ou) — Booster Energy]: the recommended set is Booster Energy with Fiery Dance / Sludge Wave / coverage / Substitute, i.e. the item is spent on entry BY DESIGN to switch on Quark Drive. Consistent with the item-polarity finding (consumption is the point, not a loss). Remaining unpriced question is TIMING — which entry spends it — which the eval does not model.

</details>

### alomomola — 8.9% usage · *Smogon-cited*

**tags** pivot, annuity, wall · **ability** regenerator · **preserve** high · **deployment** pivot-cycle

Regenerator restores HP on switch out and Wish heals a teammate, so it is a healing source for the whole side rather than only itself. Flip Turn pivots.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Alomomola (gen9ou) — WishFish (Wish Passer)]: Recommended set is "Flip Turn; Wish; Protect; Scald / Tickle" with Regenerator. TEAM-WIDE HEALING ANNUITY: Wish heals a TEAMMATE, and Regenerator heals itself on every switch — so its survival is a recurring income stream for the whole side, the same economics as Poison Heal Gliscor. Directly relevant to the recovery war measured in the long grinds.

</details>

### darkrai — 8.1% usage · *Smogon-cited*

**tags** revenge-killer, wallbreaker · **ability** baddreams · **preserve** med · **deployment** late-cleaner

Very high Speed, usually with Choice Scarf, makes it a revenge killer. Trick hands the Choice item to a wall to cripple it.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Darkrai (gen9ou) — Choice Scarf]: "Darkrai can also cripple physical walls like Dondozo and Skarmory with Trick." Trick converts its own item into a liability on their wall — a resource TRADE the eval scores as an item swap. Also the species our speed-floor inference must not mistake for a Scarf when it is not (see the 88f1f17 correction).

</details>

### walkingwake — 7.3% usage · *measured here*

**tags** weather-abuser, sweeper · **ability** protosynthesis · **preserve** med · **requires** sun

Hydro Steam is strengthened rather than weakened in sun. Protosynthesis activates in sun or from Booster Energy, so it still functions when the sun is gone.

<details><summary>provenance (review only — not shown to any consumer)</summary>

Hydro Steam is sun-boosted rather than sun-nerfed, and Protosynthesis falls back on Booster Energy when the sun drops — which is exactly WHY sun tolerates engine piloting (63%) while rain does not (29.2%, benched). Recorded to keep the weather fix from treating all abusers as equally weather-dependent.

</details>

### rillaboom — 7.3% usage · *Smogon-cited*

**tags** terrain-setter · **ability** grassysurge · **preserve** med · **resource** grassyterrain

Grassy Surge sets Grassy Terrain, which heals every grounded Pokemon each turn and weakens Earthquake — a benefit to its whole side that ends when it does. Grassy Glide gains priority in terrain.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Rillaboom (gen9ou) — Utility]: Grassy Terrain is described as "solving its problem with longevity" for partners and enabling Grassy-Seed setup — i.e. the terrain is a TEAM-WIDE annuity (passive recovery + halved Earthquake), not a personal buff, so the setter's death ends a resource the whole side was drawing on. Same economics shape as the weather setters; RE-RUN the rain_audit for terrain before assigning preserve pressure, since sun leaked identically to rain and did not care.

</details>

### zapdos — 7.0% usage · *Smogon-cited*

**tags** pivot · **ability** static · **preserve** med · **deployment** pivot-cycle

Volt Switch pivots while Roost restores HP, so it re-enters many times a game. Heavy-Duty Boots let it ignore entry hazards while doing so.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Zapdos (gen9ou) — Offensive]: Recommended set "Hurricane; Volt Switch; Heat Wave; Roost" with Heavy-Duty Boots: "a very proactive Pokemon that can help gain momentum." Boots + Roost = a mon designed to re-enter repeatedly, so its value is in the switch economy the hazard war taxes.

</details>

### deoxysspeed — 6.3% usage · *Smogon-cited*

**tags** suicide-lead, screens-setter, hazard-setter · **preserve** low · **deployment** lead · **lead_intent** strong

Extreme Speed stat lets it set hazards or screens before almost anything else moves. Focus Sash guarantees it acts at least twice.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Deoxys-Speed (gen9ou) — Hazard Lead]: "Focus Sash guarantees it can set at least one layer of entry hazard against faster threats... or even set two layers against slower threats" — the set is explicitly built to trade itself for hazards, confirming the suicide-lead role rather than a mon to preserve.

</details>

### garganacl — 6.2% usage · *Smogon-cited*

**tags** wall, wincon · **ability** purifyingsalt · **preserve** high · **deployment** setup-window

Purifying Salt blocks status. Iron Defense with Body Press converts its Defense into attacking power, while Salt Cure chips and Recover sustains — a slow plan that needs several turns to pay off.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Garganacl (gen9ou) — Iron Defense]: "With fantastic bulk, Salt Cure, Purifying Salt's status immunity, the combination of Iron Defense" and Body Press + Recover. IronPress is a slow multi-turn wincon whose intermediate states look flat — the exact accumulation pattern the leaf eval prunes (the documented Curse/Dondozo failure mode). Salt Cure is chip we measured at only 1% of damage taken, so its threat here is the setup, not the chip.

</details>

### pelipper — 4.7% usage · *measured here*

**tags** weather-setter · **ability** drizzle · **preserve** high · **deployment** lead · **lead_intent** strong · **resource** rain

Drizzle sets rain the moment it enters, and Damp Rock extends the duration. The rain ends early if another weather setter replaces it.

<details><summary>provenance (review only — not shown to any consumer)</summary>

rain_audit over 27 rain-team games: 43% average uptime where a piloted Damp Rock team lives 60-80%; uptime tracked outcomes (50% in wins vs 40% in losses). Rain teams benched at 29.2% board-only, the only below-pool-CI archetype.

</details>

### torkoal — 3.3% usage · *measured here*

**tags** weather-setter · **ability** drought · **preserve** med · **deployment** lead · **lead_intent** strong · **resource** sun

Drought sets sun on entry and Heat Rock extends it. Its own Speed is very low, so it exists to enable partners.

<details><summary>provenance (review only — not shown to any consumer)</summary>

same audit: sun leaks identically (43% uptime) but sun teams win anyway (63%) because Protosynthesis abusers fall back on Booster Energy — so preserve pressure is genuinely LOWER here than for rain. Recorded to stop a future fix over-generalising 'weather setter' into one rule.

</details>

### barraskewda — 2.6% usage · *measured here*

**tags** weather-abuser, sweeper · **ability** swiftswim · **preserve** med · **value_curve** decays_with_weather_clock · **requires** rain

Swift Swim doubles its Speed in rain, so it outspeeds most of the field only while rain is up and is ordinary without it.

<details><summary>provenance (review only — not shown to any consumer)</summary>

41% of Swift Swim moves across those games were clicked OUTSIDE rain (60 in / 42 out) — the search spends the abuser while its enabling condition is down.

</details>

### grimmsnarl — 1.8% usage · *Smogon-cited*

**tags** suicide-lead, screens-setter · **ability** prankster · **preserve** low · **deployment** lead · **lead_intent** strong

Prankster gives its status moves priority, so it sets Reflect and Light Screen before being attacked. Light Clay extends both to eight turns.

<details><summary>provenance (review only — not shown to any consumer)</summary>

RAG-grounded [smogon#Grimmsnarl (gen9ou) — Dual Screens]: recommended set is Reflect / Light Screen / Taunt / Spirit Break with LIGHT CLAY, an item bought purely to extend the side conditions it leaves behind — the definition of expendable team-persistent support. Matches the monotype screens-setter precedent.

</details>

## Known schema gap

`preserve` for a hazard remover (Great Tusk, Corviknight) depends on whether **our own** team has another remover — a team property this per-species file cannot express. `conditional` only covers opponent facts. Needs a team-level pass before anything consumes it.
