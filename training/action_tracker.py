# ActionTrackerCallback: logs per-step action category distributions + episode rewards
# Usage: add to SB3 callback list alongside eval/self-play callbacks

from __future__ import annotations

from collections import deque

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

CATEGORIES = ("damage", "status", "setup", "other", "switch", "forced_switch", "struggle")

# action sequences worth tracking (core interactions)
# all 25 action sequences (5x5 grid)
#
# damage -> damage:  sustained aggression, trying to KO
# damage -> status:  soften up then cripple (para/sleep after chip)
# damage -> setup:   chip then boost for the sweep
# damage -> other:   hit then recover/weather (momentum pause)
# damage -> switch:  hit then reposition (scouting or pivot)
#
# status -> damage:  cripple then capitalize (para then sweep)
# status -> status:  stacking ailments (toxic + confuse, para + leech seed)
# status -> setup:   cripple then set up safely (para then swords dance)
# status -> other:   ailment then sustain (toxic then recover stall)
# status -> switch:  status then pivot (toxic then switch to counter)
#
# setup -> damage:   the payoff -- boost then attack (SD then sweep)
# setup -> status:   boost then cripple (unusual, maybe curse then toxic)
# setup -> setup:    double boost (SD + agility, curse + curse)
# setup -> other:    boost then sustain (curse then rest)
# setup -> switch:   boost then baton pass, or aborted setup
#
# other -> damage:   recover/weather then attack (rain dance then surf)
# other -> status:   recover then cripple (rest talk into status?)
# other -> setup:    weather then boost (rain dance then swords dance)
# other -> other:    double sustain (recover twice, stalling)
# other -> switch:   recover then pivot (heal up then switch out)
#
# switch -> damage:  reposition then attack (bring in counter and hit)
# switch -> status:  reposition then cripple (bring in statuser)
# switch -> setup:   reposition then boost (bring in sweeper to set up)
# switch -> other:   reposition then sustain (switch to wall, recover)
# switch -> switch:  double switch (repositioning)
#
# forced_switch -> *: what the agent does after a KO (reveals follow-up priorities)
# * -> forced_switch: not meaningful (KO is caused by opponent, not agent choice)
_SEQ_CATS = ("damage", "status", "setup", "other", "switch", "forced_switch")
_ABBREV = {"forced_switch": "fsw", "switch": "swi", "damage": "dam",
           "status": "sta", "setup": "set", "other": "oth", "struggle": "str"}
SEQUENCES = tuple(
    (a, b)
    for a in _SEQ_CATS
    for b in _SEQ_CATS
)


class ActionTrackerCallback(BaseCallback):
    """Track action type distribution, sequences, and episode rewards."""

    def __init__(self, log_freq: int = 10_000, window: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._window: deque[str] = deque(maxlen=window)
        self._ep_rewards: deque[float] = deque(maxlen=500)
        self._ep_accum: np.ndarray | None = None  # per-env running totals
        self._prev_action: list[str | None] | None = None  # per-env previous action
        self._seq_counts: dict[tuple[str, str], int] = {s: 0 for s in SEQUENCES}
        self._seq_total = 0
        self._last_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        # init per-env state on first step
        if self._ep_accum is None:
            self._ep_accum = np.zeros(len(infos), dtype=np.float64)
            self._prev_action = [None] * len(infos)

        for i, info in enumerate(infos):
            atype = info.get("action_type")
            if atype:
                self._window.append(atype)
                # track sequences
                prev = self._prev_action[i]
                if prev is not None:
                    pair = (prev, atype)
                    if pair in self._seq_counts:
                        self._seq_counts[pair] += 1
                    self._seq_total += 1
                self._prev_action[i] = atype

        # accumulate rewards and detect episode boundaries
        for i in range(len(rewards)):
            self._ep_accum[i] += rewards[i]
            if dones[i]:
                self._ep_rewards.append(float(self._ep_accum[i]))
                self._ep_accum[i] = 0.0
                self._prev_action[i] = None  # reset on episode boundary

        if self.num_timesteps - self._last_log >= self.log_freq and len(self._window) > 0:
            self._last_log = self.num_timesteps
            total = len(self._window)
            counts = {c: 0 for c in CATEGORIES}
            for a in self._window:
                counts[a] = counts.get(a, 0) + 1

            parts = []
            for cat in CATEGORIES:
                pct = counts[cat] / total
                self.logger.record(f"actions/{cat}_pct", pct)
                parts.append(f"{cat}={pct:.1%}")

            if self._ep_rewards:
                rews = np.array(self._ep_rewards)
                mean = float(rews.mean())
                std = float(rews.std())
                self.logger.record("reward/ep_mean", mean)
                self.logger.record("reward/ep_std", std)

            # log sequence percentages
            seq_parts = []
            if self._seq_total > 0:
                for seq in SEQUENCES:
                    pct = self._seq_counts[seq] / self._seq_total
                    label = f"{_ABBREV.get(seq[0], seq[0][:3])}>{_ABBREV.get(seq[1], seq[1][:3])}"
                    self.logger.record(f"sequences/{seq[0]}_{seq[1]}", pct)
                    seq_parts.append(f"{label}={pct:.1%}")
                # reset for next window
                self._seq_counts = {s: 0 for s in SEQUENCES}
                self._seq_total = 0

            if self.verbose:
                print(f"  [{self.num_timesteps:>10,}] actions: {', '.join(parts)}")
                if self._ep_rewards:
                    print(f"  [{self.num_timesteps:>10,}] reward:  mean={mean:.3f}, std={std:.3f} (n={len(self._ep_rewards)})")
                if seq_parts:
                    print(f"  [{self.num_timesteps:>10,}] chains:  {', '.join(seq_parts)}")

        return True
