# Role annotations — review copy

Generated from `showdown/roles.json` (34 entries). **Edit the JSON, not this file.**

Nothing consumes these yet. Every entry carries `evidence` and a review grade; an entry whose claim cannot be traced is a guess and should not ship.

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
| 21.9% | **gholdengo** ⚡ | measured here | med | — | spinblocker |
| 17.9% | **dragonite** | Smogon-cited | high | setup-window | wincon, setup-sweeper |
| 15.3% | **ironvaliant** | Smogon-cited | med | — | wallbreaker, setup-sweeper |
| 15.3% | **ragingbolt** | Smogon-cited | med | late-cleaner | wallbreaker, priority-attacker |
| 15.2% | **zamazenta** | Smogon-cited | high | — | wall, wincon, glue |
| 15.1% | **ogerponwellspring** | Smogon-cited | med | — | wallbreaker, sweeper |
| 14.6% | **dragapult** | Smogon-cited | med | pivot-cycle | pivot, wallbreaker |
| 12.9% | **hatterene** ⚡ | USER-CORRECTED | med | bait-switch | wall, sacrificial-support |
| 12.8% | **corviknight** ⚡ | Smogon-cited | med | pivot-cycle | hazard-removal, pivot, wall |
| 12.5% | **slowkinggalar** | measured here | med | pivot-cycle | pivot, wall |
| 11.2% | **gliscor** | measured here | high | pivot-cycle | annuity, wall |
| 11.0% | **irontreads** | Smogon-cited | med | lead | hazard-removal, hazard-setter, lead |
| 10.6% | **kyurem** | Smogon-cited | med | — | wallbreaker |
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

⚡ = value depends on the opponent's team ▸ = has a written play sequence

## Entries

### greattusk — 32.9% usage · *Smogon-cited*

**tags** hazard-removal, wall, pivot · **ability** protosynthesis · **preserve** med

**Conditional — `opponent_has:hazard-setter`:** preserve → high

> Removal is only a team-persistent resource when they are actually setting. Still carries the unresolved SOLE-REMOVER caveat: value also depends on whether OUR team has another remover, a team property this per-species file cannot express.

<details><summary>evidence</summary>

RAG-grounded [smogon#Great Tusk (gen9ou) — Offensive Utility / Defensive]: "Rapid Spin allows it to remove entry hazards for its teammates, and the Speed boost lets it scare out usually faster threats like Darkrai and Iron Crown"; also "a staple on Sticky Web teams". SOLE-REMOVER CAVEAT: preserve:high holds only when it is the team's only remover — a TEAM property this per-species file cannot express (schema gap). Hazard chip is our pinned loss mechanism (SR = 21% of damage taken), so the remover is a team-persistent resource.

</details>

### kingambit — 22.5% usage · *measured here*

**tags** cleaner, wincon · **ability** supremeoverlord · **preserve** high · **deployment** late-cleaner · **lead_intent** avoid · **value_curve** grows_with_own_faints

<details><summary>evidence</summary>

Supreme Overlord = +10% power per fallen ally (abilities.rs:2257), now mirrored into threat_vs (c68f15a). Timing instrument: we first deploy it ~T9 with 0.44 allies down and 77% of the time with ZERO down, vs the ladder population's T13.0 / 0.96. fp is equally guilty, so this is absolute strength, not fp-gap.

</details>

### gholdengo — 21.9% usage · *measured here*

**tags** spinblocker · **ability** goodasgold · **preserve** med

**Conditional — `opponent_has:hazard-removal`:** preserve → high · tags → ['spinblocker']

> Spinblocking is worth nothing against a team with no remover. richwoman pairs it with 3+ SR setters, which is exactly when it is load-bearing.

<details><summary>evidence</summary>

richwoman's hazard-stack identity runs Gholdengo as the spinblock behind 3+ SR setters (11/23 of her games); its survival is what makes her chip mechanism work — the mirror of our own removal dependency.

</details>

### dragonite — 17.9% usage · *Smogon-cited*

**tags** wincon, setup-sweeper · **ability** multiscale · **preserve** high · **deployment** setup-window · **entry_condition** full_hp · **value_curve** decays_with_chip

<details><summary>evidence</summary>

RAG-grounded [smogon#Dragonite (gen9ou)]: Roost-based sets "give Dragonite easy setup opportunities" and let it check Ogerpon-W, Iron Moth and Rillaboom, "with Multiscale intact". Same entry_condition family as Ceruledge: Multiscale only halves damage at FULL HP, so entry chip deletes the defensive half of the wincon before it sets up — which is why the standard set runs Boots. The eval prices Multiscale when at full HP but cannot price PRESERVING full HP for a later sweep.

</details>

### ironvaliant — 15.3% usage · *Smogon-cited*

**tags** wallbreaker, setup-sweeper · **ability** quarkdrive · **preserve** med

<details><summary>evidence</summary>

RAG-grounded [smogon#Iron Valiant (gen9ou)]: "Choice Specs takes advantage of Moonblast's spammability and lets Iron Valiant break through and even 2HKO otherwise common Iron Valiant answers such as Assault Vest Hatterene, Galarian Weezing, and specially defensive Gliscor." Booster Energy is its top item by usage — a one-shot entry resource like Iron Moth's, so entry TIMING is the unpriced question.

</details>

### ragingbolt — 15.3% usage · *Smogon-cited*

**tags** wallbreaker, priority-attacker · **ability** protosynthesis · **preserve** med · **deployment** late-cleaner

<details><summary>evidence</summary>

RAG-grounded [smogon#Raging Bolt (gen9ou) — Choice Specs]: "Thunderclap is used as powerful priority to revenge kill or force out faster threats such as Ogerpon-W, Iron Valiant, and Enamorus, and it also helps Raging Bolt beat slower priority users such as Kingambit and Scizor." Priority-in-reserve is a role the leaf eval sees only as a weak move until the turn it matters — same shape as Ceruledge's Shadow Sneak.

</details>

### zamazenta — 15.2% usage · *Smogon-cited*

**tags** wall, wincon, glue · **ability** dauntlessshield · **preserve** high

<details><summary>evidence</summary>

RAG-grounded [smogon#Zamazenta (gen9ou)]: "Zamazenta is the tier's most consistent glue piece and wincon. Its natural bulk alongside a valuable typing, which can be boosted further by Dauntless Shield, lets it take on many manner of offensive threats like Kingambit, Hisuian Samurott, some Dragonite sets, and Great Tusk." Glue = it is what other members are built around; losing it costs more than its own HP.

</details>

### ogerponwellspring — 15.1% usage · *Smogon-cited*

**tags** wallbreaker, sweeper · **ability** waterabsorb · **preserve** med

<details><summary>evidence</summary>

RAG-grounded [smogon#Ogerpon-Wellspring (gen9ou)]: "one of OU's premier physical attackers" with "the high power of Ivy Cudgel and Power Whip, the option to trade the latter with Trailblaze to outspeed otherwise faster foes". Trailblaze makes it a conditional sweeper — a setup curve the eval prices only one turn at a time.

</details>

### dragapult — 14.6% usage · *Smogon-cited*

**tags** pivot, wallbreaker · **ability** infiltrator · **preserve** med · **deployment** pivot-cycle

<details><summary>evidence</summary>

RAG-grounded [smogon#Dragapult (gen9ou) — Boots Pivot]: "Infiltrator is the preferred ability, as it allows Dragapult to both hit and cripple foes regardless of" substitutes and screens. Infiltrator ignoring Substitute/screens is a matchup fact the eval does model; the role note is that it is the momentum piece, not a mon to hold back.

</details>

### hatterene — 12.9% usage · *USER-CORRECTED*

**tags** wall, sacrificial-support · **ability** magicbounce · **preserve** med · **deployment** bait-switch · **lead_intent** avoid

**Conditional — `opponent_has:hazard-setter`:** preserve → high · tags → ['hazard-denial', 'wall', 'sacrificial-support']

> Magic Bounce only has a job if they have hazards to bounce. Against a setter-carrying team it is the anchor of the hazard war and its death re-opens the SR channel that costs us 21% of damage taken; against a team without setters the same mon is a bulky attacker and nothing more.

> NOT a lead — it is switched in on a PREDICTED hazard turn, baiting the setter into clicking Stealth Rock or Spikes so the layer lands on their own side. The bait is the point, so a conservative pilot that only brings it in when safe never collects the value.

<details><summary>evidence</summary>

RAG-grounded [smogon#Hatterene (gen9ou) — Healing Wish]: Magic Bounce denies hazards passively while alive; "Psychic Noise notably lets Hatterene significantly chip bulky foes like Gliscor, Garganacl, and Clefable, preventing them from recovering their health", and the Healing Wish set trades itself to "pivot them in safely". TWO team-persistent effects the eval cannot price: (a) Magic Bounce is hazard PREVENTION, worth most against exactly the hazard-stack that beats us — its death re-opens the SR war; (b) Healing Wish is a deliberate SACRIFICE that restores a teammate, so a low-HP Hatterene is not obviously a liability. Psychic Noise blocking recovery also matters in the long grinds where the recovery war is even. CONDITIONAL + DEPLOYMENT added 2026-07-31 (user): the static schema could not express that Magic Bounce's worth depends on the OPPONENT's roster, nor that the mon is played as a setup-baiting switch-in rather than a lead.

</details>

### corviknight — 12.8% usage · *Smogon-cited*

**tags** hazard-removal, pivot, wall · **ability** pressure · **preserve** med · **deployment** pivot-cycle

**Conditional — `opponent_has:hazard-setter`:** preserve → high

> Removal is only a team-persistent resource when they are actually setting. Still carries the unresolved SOLE-REMOVER caveat: value also depends on whether OUR team has another remover, a team property this per-species file cannot express.

<details><summary>evidence</summary>

RAG-grounded [smogon#Corviknight (gen9ou) — Defensive]: "Defog allows Corviknight to fulfill the role as the team's hazard remover", and Pressure "more quickly stall[s] out their attacks' PP" against setup sweepers (Dragonite, Gliscor, Kingambit). Sole-remover caveat still applies: preserve pressure is a TEAM property this per-species file cannot express. PP-stalling matters in the long grinds where the stall audit measured real PP bankruptcy.

</details>

### slowkinggalar — 12.5% usage · *measured here*

**tags** pivot, wall · **ability** regenerator · **preserve** med · **deployment** pivot-cycle · **lead_intent** neutral

<details><summary>evidence</summary>

Regenerator pivots are net-positive on entry (fp routes hazard cycles through them: pays 12.5%, regains 33%) — the hazard_cycle finding. Also the mon whose speed floor produced the false Choice Scarf call (88f1f17), so its canonical spread is bulky with ~0 Speed.

</details>

### gliscor — 11.2% usage · *measured here*

**tags** annuity, wall · **ability** poisonheal · **preserve** high · **deployment** pivot-cycle · **value_curve** grows_with_own_status

<details><summary>evidence</summary>

The flagship annuity case. Stall audit: fp's Poison Heal Gliscor played its FULL PP budget and generated 22.8 mons of free healing over three marathons while ours clicked 6 moves in ~711 turns. Eval terms shipped for exactly this (poke-engine 76af1e9: POISON_HEAL_STATUSED 15->35, PENDING +15) and our Gliscor now SubToxes at full budget. Known race: the naked orb walked into Knock Off on entry in 2/3 marathons, so activation timing is itself the play.

</details>

### irontreads — 11.0% usage · *Smogon-cited*

**tags** hazard-removal, hazard-setter, lead · **ability** quarkdrive · **preserve** med · **deployment** lead · **lead_intent** strong

<details><summary>evidence</summary>

RAG-grounded [smogon#Iron Treads (gen9ou) — Lead]: The Lead set trades itself for hazard control and "gives it another way to block Defog and Rapid Spin"; coverage choice is explicitly about stopping Raging Bolt boosting or "Pecharunt from blocking Rapid Spin". Dual hazard role (sets AND removes) makes it a lead by design rather than a mon to preserve.

</details>

### kyurem — 10.6% usage · *Smogon-cited*

**tags** wallbreaker · **ability** pressure · **preserve** med

<details><summary>evidence</summary>

RAG-grounded [smogon#Kyurem (gen9ou)]: "one of SV OU's strongest offensive Pokemon, utilizing its powerful mixed offensive stats and solid overall bulk to run a wide variety of sets, each requiring different counterplay. Choice Specs makes Kyurem into a wallbreaker." 'Each set requires different counterplay' is a direct statement that set inference matters more than usual here — relevant to the belief-tier work.

</details>

### samurotthisui — 10.3% usage · *Smogon-cited*

**tags** hazard-setter, wallbreaker · **ability** sharpness · **preserve** med · **deployment** lead · **lead_intent** strong

**Conditional — `opponent_has:hazard-denial`:** preserve → high

> Ceaseless Edge sets Spikes THROUGH Magic Bounce and Taunt, so against a Hatterene/Deoxys-style denial team it is the only setter that still functions — the counter to the counter.

<details><summary>evidence</summary>

RAG-grounded [smogon#Samurott-Hisui (gen9ou)]: "one of the most prolific entry hazard setters in the tier with its signature move Ceaseless Edge, letting it both set Spikes and deal heavy damage thanks to its Sharpness ability. It can even do so in the face of Hatterene's Magic Bounce and Taunt." Setting hazards THROUGH Magic Bounce is the counter to hazard-denial — the other side of the war that decides our long games.

</details>

### cinderace — 10.0% usage · *Smogon-cited*

**tags** pivot, hazard-control · **ability** libero · **preserve** med

<details><summary>evidence</summary>

RAG-grounded [smogon#Cinderace (gen9ou) — Offensive Pivot]: Recommended set is "Pyro Ball; U-turn; Court Change; ..." with Heavy-Duty Boots and Libero. COURT CHANGE is hazard control by THEFT — it flips our hazards onto them rather than clearing them, which is strictly better in a chip war and is why our ah3 team runs it. The eval sees a side-condition swap but not that it converts their investment into ours.

</details>

### tinglu — 9.6% usage · *measured here*

**tags** hazard-setter, wall · **ability** vesselofruin · **preserve** med · **deployment** lead · **lead_intent** strong

<details><summary>evidence</summary>

richwoman's most-used lead by a distance (45 of her games) and the anchor of the hazard-stack that produces our pinned chip loss. Her revealed set is SR/Whirlwind/Earthquake/Ruination — Whirlwind is what converts her hazards into forced-switch chip.

</details>

### landorustherian — 9.3% usage · *Smogon-cited*

**tags** pivot, wall, glue · **ability** intimidate · **preserve** high · **deployment** pivot-cycle

<details><summary>evidence</summary>

RAG-grounded [smogon#Landorus-Therian (gen9ou)]: "one of OU's premier pivots. It can check a wide variety of threats such as Kingambit, Ceruledge, and Raging Bolt." Intimidate is a persistent team-wide defensive effect applied on every entry — an annuity the eval prices only as a one-off stat drop.

</details>

### ceruledge — 9.2% usage · *USER-CORRECTED*

**tags** setup-sweeper, wincon · **ability** weakarmor · **preserve** med · **deployment** setup-window · **entry_condition** full_hp · **value_curve** decays_with_chip

**The play:**

1. enter at FULL HP on a free switch — Focus Sash only functions from full, so entry chip cancels the plan
2. Swords Dance while they attack or switch
3. TAKE the hit: Sash holds at 1 HP, which procs Weak Armor for +Speed (losing the Sash here is the mechanism, not a loss)
4. now outspeeding: attack with Bitter Blade, whose drain heals back off 1 HP
5. hold Shadow Sneak in reserve for anything that still outspeeds or carries priority

> Every intermediate state scores badly in isolation — 1 HP, no item, minus Defense — and only the completed chain is a sweep. Same failure family as the Curse/Dondozo accumulation the eval prunes, and as Hatterene's bait-switch: the correct play looks locally unsafe.

<details><summary>evidence</summary>

TWICE-CORRECTED, and the final read is the user's (2026-07-31). (1) My inference: 'Focus Sash is a resource chip destroys, so preserve it.' (2) RAG partly corrected it [smogon#Ceruledge (gen9ou) — Swords Dance]: popping the Sash with weak moves "would only give it a trouble-free Weak Armor boost", so the opponent removing the Sash is often GOOD for Ceruledge — I then over-corrected to preserve:low. (3) The real mechanism is a SEQUENCE, and entry chip is what breaks it: come in at FULL HP (Focus Sash only functions from full), Swords Dance, TAKE a hit, survive at 1 HP on Sash, which procs Weak Armor for +Speed, then outspeed and heal back with Bitter Blade's drain, holding Shadow Sneak in reserve for anything that still outspeeds or carries priority. So the Sash is not a thing to protect and not a thing to spend — it is the PIVOT of a multi-turn plan whose precondition is an uncontested entry. Any Stealth Rock chip kills the plan before turn one. DIRECTLY TIED to our pinned loss mechanism (SR = 21% of damage taken in the long grinds): this is a mon whose wincon the hazard war silently deletes, which is also why the standard set wants removal or Boots support. The whole sequence is exactly the beyond-horizon multi-turn plan the leaf eval cannot price.

</details>

### pecharunt — 9.2% usage · *measured here*

**tags** wall, status-spreader · **ability** poisonpuppeteer · **preserve** med

<details><summary>evidence</summary>

CORRECTION 2026-07-31 (user-flagged, verified): Poison Puppeteer IS implemented — generate_instructions.rs:907 adds CONFUSION to a foe it poisons, landed in 2e000e6 alongside Rock Head/Anger Shell/Synchronize. An earlier memory note listing it as a silent OU gap was stale. The engine models it correctly; this is now an ordinary role entry.

</details>

### glimmora — 9.0% usage · *Smogon-cited*

**tags** suicide-lead, hazard-setter · **preserve** low · **deployment** lead · **lead_intent** strong

<details><summary>evidence</summary>

RAG-grounded [smogon#Glimmora (gen9ou) — Lead]: the Lead set exists to deny setup — it stops "Ting-Lu from setting up entry hazards multiple times and Hatterene from trying to deny Stealth Rock with Magic Bounce". Confirms the suicide-lead role: its job is completed on the turns it survives, not by surviving. Directly relevant to richwoman, whose hazard stack is anchored by Ting-Lu (45 leads).

</details>

### ironmoth — 8.9% usage · *Smogon-cited*

**tags** setup-sweeper · **ability** quarkdrive · **preserve** med

<details><summary>evidence</summary>

RAG-grounded [smogon#Iron Moth (gen9ou) — Booster Energy]: the recommended set is Booster Energy with Fiery Dance / Sludge Wave / coverage / Substitute, i.e. the item is spent on entry BY DESIGN to switch on Quark Drive. Consistent with the item-polarity finding (consumption is the point, not a loss). Remaining unpriced question is TIMING — which entry spends it — which the eval does not model.

</details>

### alomomola — 8.9% usage · *Smogon-cited*

**tags** pivot, annuity, wall · **ability** regenerator · **preserve** high · **deployment** pivot-cycle

<details><summary>evidence</summary>

RAG-grounded [smogon#Alomomola (gen9ou) — WishFish (Wish Passer)]: Recommended set is "Flip Turn; Wish; Protect; Scald / Tickle" with Regenerator. TEAM-WIDE HEALING ANNUITY: Wish heals a TEAMMATE, and Regenerator heals itself on every switch — so its survival is a recurring income stream for the whole side, the same economics as Poison Heal Gliscor. Directly relevant to the recovery war measured in the long grinds.

</details>

### darkrai — 8.1% usage · *Smogon-cited*

**tags** revenge-killer, wallbreaker · **ability** baddreams · **preserve** med · **deployment** late-cleaner

<details><summary>evidence</summary>

RAG-grounded [smogon#Darkrai (gen9ou) — Choice Scarf]: "Darkrai can also cripple physical walls like Dondozo and Skarmory with Trick." Trick converts its own item into a liability on their wall — a resource TRADE the eval scores as an item swap. Also the species our speed-floor inference must not mistake for a Scarf when it is not (see the 88f1f17 correction).

</details>

### walkingwake — 7.3% usage · *measured here*

**tags** weather-abuser, sweeper · **ability** protosynthesis · **preserve** med · **requires** sun

<details><summary>evidence</summary>

Hydro Steam is sun-boosted rather than sun-nerfed, and Protosynthesis falls back on Booster Energy when the sun drops — which is exactly WHY sun tolerates engine piloting (63%) while rain does not (29.2%, benched). Recorded to keep the weather fix from treating all abusers as equally weather-dependent.

</details>

### rillaboom — 7.3% usage · *Smogon-cited*

**tags** terrain-setter · **ability** grassysurge · **preserve** med · **resource** grassyterrain

<details><summary>evidence</summary>

RAG-grounded [smogon#Rillaboom (gen9ou) — Utility]: Grassy Terrain is described as "solving its problem with longevity" for partners and enabling Grassy-Seed setup — i.e. the terrain is a TEAM-WIDE annuity (passive recovery + halved Earthquake), not a personal buff, so the setter's death ends a resource the whole side was drawing on. Same economics shape as the weather setters; RE-RUN the rain_audit for terrain before assigning preserve pressure, since sun leaked identically to rain and did not care.

</details>

### zapdos — 7.0% usage · *Smogon-cited*

**tags** pivot · **ability** static · **preserve** med · **deployment** pivot-cycle

<details><summary>evidence</summary>

RAG-grounded [smogon#Zapdos (gen9ou) — Offensive]: Recommended set "Hurricane; Volt Switch; Heat Wave; Roost" with Heavy-Duty Boots: "a very proactive Pokemon that can help gain momentum." Boots + Roost = a mon designed to re-enter repeatedly, so its value is in the switch economy the hazard war taxes.

</details>

### deoxysspeed — 6.3% usage · *Smogon-cited*

**tags** suicide-lead, screens-setter, hazard-setter · **preserve** low · **deployment** lead · **lead_intent** strong

<details><summary>evidence</summary>

RAG-grounded [smogon#Deoxys-Speed (gen9ou) — Hazard Lead]: "Focus Sash guarantees it can set at least one layer of entry hazard against faster threats... or even set two layers against slower threats" — the set is explicitly built to trade itself for hazards, confirming the suicide-lead role rather than a mon to preserve.

</details>

### garganacl — 6.2% usage · *Smogon-cited*

**tags** wall, wincon · **ability** purifyingsalt · **preserve** high · **deployment** setup-window

<details><summary>evidence</summary>

RAG-grounded [smogon#Garganacl (gen9ou) — Iron Defense]: "With fantastic bulk, Salt Cure, Purifying Salt's status immunity, the combination of Iron Defense" and Body Press + Recover. IronPress is a slow multi-turn wincon whose intermediate states look flat — the exact accumulation pattern the leaf eval prunes (the documented Curse/Dondozo failure mode). Salt Cure is chip we measured at only 1% of damage taken, so its threat here is the setup, not the chip.

</details>

### pelipper — 4.7% usage · *measured here*

**tags** weather-setter · **ability** drizzle · **preserve** high · **deployment** lead · **lead_intent** strong · **resource** rain

<details><summary>evidence</summary>

rain_audit over 27 rain-team games: 43% average uptime where a piloted Damp Rock team lives 60-80%; uptime tracked outcomes (50% in wins vs 40% in losses). Rain teams benched at 29.2% board-only, the only below-pool-CI archetype.

</details>

### torkoal — 3.3% usage · *measured here*

**tags** weather-setter · **ability** drought · **preserve** med · **deployment** lead · **lead_intent** strong · **resource** sun

<details><summary>evidence</summary>

same audit: sun leaks identically (43% uptime) but sun teams win anyway (63%) because Protosynthesis abusers fall back on Booster Energy — so preserve pressure is genuinely LOWER here than for rain. Recorded to stop a future fix over-generalising 'weather setter' into one rule.

</details>

### barraskewda — 2.6% usage · *measured here*

**tags** weather-abuser, sweeper · **ability** swiftswim · **preserve** med · **value_curve** decays_with_weather_clock · **requires** rain

<details><summary>evidence</summary>

41% of Swift Swim moves across those games were clicked OUTSIDE rain (60 in / 42 out) — the search spends the abuser while its enabling condition is down.

</details>

### grimmsnarl — 1.8% usage · *Smogon-cited*

**tags** suicide-lead, screens-setter · **ability** prankster · **preserve** low · **deployment** lead · **lead_intent** strong

<details><summary>evidence</summary>

RAG-grounded [smogon#Grimmsnarl (gen9ou) — Dual Screens]: recommended set is Reflect / Light Screen / Taunt / Spirit Break with LIGHT CLAY, an item bought purely to extend the side conditions it leaves behind — the definition of expendable team-persistent support. Matches the monotype screens-setter precedent.

</details>

## Known schema gap

`preserve` for a hazard remover (Great Tusk, Corviknight) depends on whether **our own** team has another remover — a team property this per-species file cannot express. `conditional` only covers opponent facts. Needs a team-level pass before anything consumes it.
