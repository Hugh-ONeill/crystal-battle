# Training Run Log

## Run 1 -- MLP baseline (reward shaping v1)
- **Arch**: MLP 128x128, n_steps=2048, batch=256
- **Obs**: 171 features (no PP frac, no damage class, no hazards/weather/status turns)
- **Reward**: HP diff + full shaping (status, faint, spikes, stat changes, switch matchup)
- **Steps**: 10M
- **Results**:
  - vs Random: ~92-95%
  - vs MaxDamage: 40-48% (plateau)
- **Action dist**: damage ~60%, status 37-41% (early), switch 9-13%
- **Notes**: Status rewards way too high, agent spammed status moves. Halved status rewards
  and added switch matchup reward (0.01) mid-run. Switch recovered to 17-19%.
  MLP hit ceiling -- can't learn multi-turn strategy without memory.

## Run 2 -- LSTM + old obs (reward shaping v2)
- **Arch**: LSTM 128x128, n_steps=256, batch=256
- **Obs**: 171 features (same as MLP)
- **Reward**: HP diff + halved status bonuses + faint + spikes + stat changes + switch matchup
- **Steps**: ~2M (killed early)
- **Results at 2M**:
  - vs Random: not evaluated (killed before eval)
- **Training metrics**: KL 0.08-0.15, clip 0.37-0.44, explained variance 0.65-0.87
- **Notes**: Training ran hot (high KL/clip) but stable. Killed to upgrade obs space.

## Run 3 -- LSTM + new obs + full shaping
- **Arch**: LSTM 128x128, n_steps=256, batch=256
- **Obs**: 234 features (added PP frac, damage class, ailment/flinch/drain/healing,
  status turns, leech seed, accuracy/evasion stages, spikes, reflect, light screen,
  weather, full opp bench stats)
- **Reward**: HP diff + halved status bonuses + faint + spikes + stat changes + switch matchup
- **Steps**: ~2M (killed after eval)
- **Results at 2M**:
  - vs Random: 77%
  - vs MaxDamage: 25%
  - Actions: damage=47.5%, status=22.6%, switch=29.8%
  - Reward mean: 1.367 (inflated by shaping farming)
- **Training metrics**: KL 0.10-0.15, clip 0.38-0.44, explained variance 0.54-0.84
- **Notes**: Richer obs made shaping farming WORSE -- agent could see statuses/hazards
  it was rewarded for and optimized to farm them. 52% of actions were non-damaging.

## Run 4 -- LSTM + new obs + minimal shaping
- **Arch**: LSTM 128x128, n_steps=256, batch=256
- **Obs**: 234 features
- **Reward**: HP diff + faint bonus only (all other shaping removed)
- **Steps**: 6M+ (killed mid-phase 2)
- **Results**:
  - 2M: vs Random 85.5%, vs MaxDamage 26.5%, damage=70.1%, status=12.4%, switch=17.4%
  - 4M: vs Random 84.0%, vs MaxDamage 30.0%, damage=65.8%, status=13.9%, switch=20.3%
  - 6M: vs Random 84.5%, vs MaxDamage 30.0%, damage=64.7%, status=19.1%, switch=16.2%
- **Training metrics**: KL 0.15-0.20, clip 0.40-0.45, entropy ~1.03
- **Notes**: Action distribution much healthier without shaping. But vs MaxDamage plateaued
  at 30% through phase 2. KL high (0.15-0.20) -- policy may be thrashing with n_steps=256.
  Status crept back to 19% by 6M. Entropy very low (1.03) = locked into bad strategy.

## Run 5 -- LSTM + new obs + minimal shaping + n_steps=2048
- **Arch**: LSTM 128x128, n_steps=2048, batch=1024
- **Obs**: 234 features
- **Reward**: HP diff + faint bonus only
- **Steps**: 7.5M (killed, plateaued)
- **Phase split**: 20% random / 40% mixed / 40% self-play (default)
- **Results**:
  - 500k: vs Random 86.0%, vs MaxDamage 22.5%
  - 1.0M: vs Random 86.5%, vs MaxDamage 27.0%
  - 1.5M: vs Random 85.5%, vs MaxDamage 30.5%
  - 2.0M: vs Random 87.0%, vs MaxDamage 27.5%  (phase 2 starts)
  - 2.5M: vs Random 87.5%, vs MaxDamage 32.0%
  - 3.0M: vs Random 90.0%, vs MaxDamage 33.0%
  - 3.5M: vs Random 86.0%, vs MaxDamage 35.0%
  - 4.5M: vs Random 91.5%, vs MaxDamage 31.0%
  - 5.0M: vs Random 86.0%, vs MaxDamage 36.0%
  - 5.5M: vs Random 86.5%, vs MaxDamage 38.0%  (peak)
  - 6.0M: vs Random 88.5%, vs MaxDamage 36.5%
  - 6.5M: vs Random 86.5%, vs MaxDamage 35.0%
  - 7.0M: vs Random 88.5%, vs MaxDamage 36.0%
  - 7.5M: vs Random 90.0%, vs MaxDamage 36.5%
- **Training metrics**: KL 0.04-0.10, clip 0.30-0.37, entropy ~1.25 (phase 1) -> ~1.08 (phase 2)
- **Notes**: Much cleaner dynamics than n_steps=256. vs MaxDamage plateaued at 35-38%,
  below MLP ceiling (40-48%). md-weight=0.5 may not give enough MaxDamage exposure.
  Phase 1 too long (2M steps vs random). Self-play not wired up (phase 3 = phase 2).

## Run 6 -- LSTM + short phase 1 + high MaxDamage weight
- **Arch**: LSTM 128x128, n_steps=2048, batch=1024
- **Obs**: 234 features
- **Reward**: HP diff + faint bonus only
- **Steps**: 9M (killed, plateaued at 37-43%)
- **Phase split**: 5% random (500k) / 40% mixed / 40% self-play (but self-play was stub)
- **md-weight**: 0.8
- **Results**:
  - 500k: vs Random 86.5%, vs MaxDamage 32.0%
  - 1.0M: vs Random 88.0%, vs MaxDamage 32.5%
  - 1.5M: vs Random 89.5%, vs MaxDamage 37.0%
  - 2.0M: vs Random 89.5%, vs MaxDamage 37.0%
  - 2.5M: vs Random 88.0%, vs MaxDamage 42.0%  (peak tied)
  - 3.0M: vs Random 91.5%, vs MaxDamage 38.0%
  - 3.5M: vs Random 85.5%, vs MaxDamage 41.0%
  - 4.0M: vs Random 84.5%, vs MaxDamage 39.0%
  - 4.5M: vs Random 86.5%, vs MaxDamage 37.0%
  - 5.0M: vs Random 87.5%, vs MaxDamage 37.5%
  - 5.5M: vs Random 85.5%, vs MaxDamage 39.5%
  - 6.0M: vs Random 84.5%, vs MaxDamage 41.0%
  - 6.5M: vs Random 89.0%, vs MaxDamage 43.5%  (peak)
  - 7.0M: vs Random 87.0%, vs MaxDamage 40.0%
  - 7.5M: vs Random 83.5%, vs MaxDamage 40.5%
  - 8.0M: vs Random 89.5%, vs MaxDamage 38.0%
  - 8.5M: vs Random 85.0%, vs MaxDamage 35.5%
  - 9.0M: vs Random 88.0%, vs MaxDamage 38.5%
- **Notes**: Matched MLP range (40-48%) but didn't exceed it. Plateaued at 37-43%.
  Shorter phase 1 + 80% MaxDamage helped reach MLP level faster. But ceiling remains
  ~40% -- likely needs real self-play to push further. Self-play was still stub (random).

## Run 7 -- LSTM + neural self-play
- **Arch**: LSTM 128x128, n_steps=2048, batch=1024
- **Obs**: 234 features
- **Reward**: HP diff + faint bonus only
- **Steps**: ~7M (killed, plateaued at same ceiling)
- **Phase split**: 500k random / 1.5M mixed (80% MD) / neural self-play + MaxDamage
- **md-weight**: 0.8
- **Self-play**: 50/50 neural vs MaxDamage per episode
- **Results**:
  - 500k: vs Random 86.0%, vs MaxDamage 32.0%
  - 1.0M: vs Random 88.0%, vs MaxDamage 28.5%
  - 1.5M: vs Random 88.0%, vs MaxDamage 41.5%
  - 2.0M: vs Random 86.0%, vs MaxDamage 34.5%  (phase 3 start)
  - 2.8M: vs Random 89.5%, vs MaxDamage 40.0%
  - 3.2M: vs Random 82.0%, vs MaxDamage 34.0%
  - 3.6M: vs Random 89.5%, vs MaxDamage 40.5%
  - 4.0M: vs Random 85.0%, vs MaxDamage 36.0%
  - 4.4M: vs Random 86.0%, vs MaxDamage 37.0%
  - 4.8M: vs Random 90.5%, vs MaxDamage 39.0%
- **Training metrics**: KL 0.07-0.15, clip 0.33-0.39, entropy 1.28 -> 1.02 (declining)
- **Notes**: Neural self-play didn't break the ~40% ceiling. Agent oscillated 34-40%
  same as without self-play. Entropy collapsed from 1.28 to 1.02 during phase 3.
  50/50 neural/MaxDamage may not give enough MaxDamage exposure. The ~40% ceiling
  appears structural -- agent can't learn type-matchup reasoning from raw features.

## Run 8 -- damage fracs + bigger net + pure MaxDamage
- **Arch**: LSTM 256x256, n_steps=2048, batch=1024
- **Obs**: 244 features (added per-move estimated damage fractions, upgraded bench
  effectiveness to damage fractions using calc_expected_damage)
- **Reward**: HP diff + faint bonus only
- **Steps**: ~5M (killed, plateaued at 40-45%)
- **Phase split**: single phase, pure MaxDamage
- **md-weight**: 1.0
- **ent_coef**: 0.05
- **Results**:
  - 0.5M: vs Random 85.5%, vs MaxDamage 37.5%
  - 1.0M: vs Random 85.5%, vs MaxDamage 42.5%
  - 1.5M: vs Random 81.5%, vs MaxDamage 44.0%
  - 2.0M: vs Random 80.0%, vs MaxDamage 45.0%  (all-time high)
  - 2.5M: vs Random 87.5%, vs MaxDamage 42.0%
  - 3.0M: vs Random 84.5%, vs MaxDamage 44.0%
  - 3.5M: vs Random 81.5%, vs MaxDamage 43.0%
  - 4.0M: vs Random 85.0%, vs MaxDamage 40.0%
  - 4.5M: vs Random 84.0%, vs MaxDamage 37.5%
  - 5.0M: vs Random 87.0%, vs MaxDamage 41.0%
- **Training metrics**: KL 0.04-0.06, clip 0.28-0.34, entropy 1.89 -> 1.50 (stable)
- **Action dist at 3M**: damage=55.5%, status=19.0%, switch=25.5%
- **Notes**: Broke old ceiling -- peaked at 45% vs MaxDamage (prev best 43.5%).
  Much more strategic play: 25% switches, switch->damage as core pattern. Damage
  fracs clearly helped. But plateaued at 40-45% oscillation band. Entropy stayed
  healthy (1.50) unlike Run 7's collapse. LR may be too high -- oscillation pattern
  suggests overshooting.

## Run 9 -- trimmed obs + KO flags + LR decay
- **Arch**: LSTM 256x256, n_steps=2048, batch=1024
- **Obs**: 198 features (removed power, stab, bench atk/def/spa/spd -- redundant with
  damage fracs. Added KO threat flags: can_i_ko, can_opp_ko_me. Added status count.)
- **Reward**: HP diff + faint bonus only
- **Steps**: ~11M (killed, oscillating 39-47%)
- **Phase split**: single phase, pure MaxDamage
- **md-weight**: 1.0
- **ent_coef**: 0.05
- **LR**: 3e-4 -> 5e-5 (linear decay)
- **Results**:
  - 0.5M: 85.5% / 40.0%   |  4.0M: 87.0% / 40.0%   |  8.0M: 85.5% / 43.5%
  - 1.0M: 90.0% / 41.0%   |  4.5M: 85.0% / 47.5%*  |  8.5M: 86.0% / 41.5%
  - 1.5M: 88.0% / 43.0%   |  5.0M: 82.5% / 43.0%   |  9.0M: 83.5% / 45.5%
  - 2.0M: 84.5% / 46.0%   |  5.5M: 84.0% / 41.0%   |  9.5M: 83.0% / 42.0%
  - 2.5M: 86.0% / 40.5%   |  6.0M: 83.0% / 39.5%   | 10.0M: 83.0% / 45.0%
  - 3.0M: 82.5% / 42.5%   |  6.5M: 85.0% / 45.0%   | 10.5M: 83.5% / 40.5%
  - 3.5M: 84.0% / 43.5%   |  7.0M: 81.0% / 45.5%   | 11.0M: 83.5% / 42.5%
  (*) all-time high: 47.5% at 4.5M
- **Notes**: New all-time high 47.5%. Average ~43%, band 39.5-47.5%. LR decay didn't
  reduce oscillation. Slightly better than Run 8 (avg ~42%, peak 45%). The ceiling
  appears to be a missing-information problem, not a convergence problem.

## Run 10 -- opponent prediction + bench survival + bigger batch (next)
- **Arch**: LSTM 256x256, n_steps=2048, batch=2048
- **Obs**: 213 features (added per-bench: opponent predicted move damage, can_ko_opp,
  survives_predicted. The predicted move = what MaxDamage will pick based on current
  active matchup. This is the actual damage a bench mon takes when switching in.)
- **Reward**: HP diff + faint bonus only
- **Steps**: 20M target
- **Phase split**: single phase, pure MaxDamage
- **md-weight**: 1.0
- **ent_coef**: 0.05
- **LR**: 3e-4 -> 5e-5 (linear decay)
- **Notes**: Key insight: current bench features show opp's best move AGAINST each bench
  mon, but switching exposes the bench mon to the move chosen against CURRENT active.
  New features tell agent the actual switch cost. Bigger batch (2048) for smoother
  gradients.
