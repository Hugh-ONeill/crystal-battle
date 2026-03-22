# PlayerState: team, active pokemon, valid actions, must_switch logic

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import Action, Forfeit, Struggle, Switch, UseMove
from .pokemon import Pokemon
from .status import SLP
from .types import TypeChart

# moves that only work when the target is asleep
_NEED_TARGET_ASLEEP = {138}  # Dream Eater
# moves that only work when the user is asleep
_NEED_USER_ASLEEP = {173, 214}  # Snore, Sleep Talk
# moves that fail when the user is already asleep
_BLOCKED_WHILE_ASLEEP = {156}  # Rest


@dataclass
class SideConditions:
    """Persistent side-of-field effects."""
    spikes: bool = False
    reflect_turns: int = 0      # 0 = inactive, counts down each turn
    light_screen_turns: int = 0


@dataclass
class PlayerState:
    team: list[Pokemon]
    active_index: int = 0
    side: SideConditions = field(default_factory=SideConditions)
    active_turns: int = 1  # consecutive turns current mon has been active

    @property
    def active(self) -> Pokemon:
        return self.team[self.active_index]

    @property
    def is_defeated(self) -> bool:
        return all(p.is_fainted for p in self.team)

    @property
    def must_switch(self) -> bool:
        """Active pokemon fainted and there are alive teammates."""
        return self.active.is_fainted and not self.is_defeated

    @property
    def alive_count(self) -> int:
        return sum(1 for p in self.team if not p.is_fainted)

    @property
    def total_hp_frac(self) -> float:
        total_max = sum(p.max_hp for p in self.team)
        total_cur = sum(p.current_hp for p in self.team)
        return total_cur / total_max if total_max > 0 else 0.0

    def valid_actions(self, opponent: "PlayerState | None" = None,
                      type_chart: TypeChart | None = None) -> list[Action]:
        """Return list of valid actions for this player."""
        actions: list[Action] = []

        if self.must_switch:
            # forced switch: only switch actions
            for i, p in enumerate(self.team):
                if i != self.active_index and not p.is_fainted:
                    actions.append(Switch(team_index=i))
            return actions

        # locked-in or recharging: force the same action, no switching
        active = self.active
        if active.recharging or active.charging_move is not None:
            # must "use" any move (the engine will handle the recharge/charge)
            actions.append(UseMove(slot_index=0))
            return actions
        if active.locked_move is not None:
            # locked into the move, find its slot
            for i, slot in enumerate(active.move_slots):
                if slot.template.id == active.locked_move.id:
                    actions.append(UseMove(slot_index=i))
                    return actions
            # fallback: shouldn't happen, but just use slot 0
            actions.append(UseMove(slot_index=0))
            return actions

        # move actions
        user_asleep = active.status == SLP
        opp_asleep = (opponent.active.status == SLP) if opponent else False
        opp_types = opponent.active.types if opponent else None
        usable = []
        immune_filtered = []
        if active.has_any_pp():
            for i, slot in enumerate(active.move_slots):
                if not slot.has_pp:
                    continue
                mid = slot.template.id
                if mid in _NEED_TARGET_ASLEEP and not opp_asleep:
                    continue
                if mid in _NEED_USER_ASLEEP and not user_asleep:
                    continue
                if mid in _BLOCKED_WHILE_ASLEEP and user_asleep:
                    continue
                move_action = UseMove(slot_index=i)
                usable.append(move_action)
                # soft-filter moves that are type-immune against opponent
                if type_chart is not None and opp_types is not None:
                    mt = slot.template
                    eff = type_chart.combined_effectiveness(mt.type, opp_types)
                    if eff == 0:
                        if mt.power > 0:
                            continue
                        meta = mt.meta or {}
                        if mt.damage_class == "status" and meta.get("ailment_id", 0) > 0:
                            continue
                immune_filtered.append(move_action)
        # prefer immune-filtered list, fall back to all usable, then struggle
        moves = immune_filtered if immune_filtered else usable
        if moves:
            actions.extend(moves)
        else:
            actions.append(Struggle())

        # switch actions: team[0]-team[5], skip active and fainted
        for i, p in enumerate(self.team):
            if i != self.active_index and not p.is_fainted:
                actions.append(Switch(team_index=i))

        return actions

    def bench_indices(self) -> list[int]:
        """Team indices of the 5 bench slots (excludes active)."""
        return [i for i in range(len(self.team)) if i != self.active_index]

    def valid_action_mask(self, opponent: "PlayerState | None" = None,
                          type_chart: TypeChart | None = None) -> list[bool]:
        """10-element mask for gym action space [move0-3, team0-5]."""
        mask = [False] * 10
        for action in self.valid_actions(opponent, type_chart=type_chart):
            if isinstance(action, UseMove):
                mask[action.slot_index] = True
            elif isinstance(action, Switch):
                # action 4+i maps directly to team[i]
                mask[4 + action.team_index] = True
            elif isinstance(action, Struggle):
                mask[0] = True  # struggle maps to action 0
        return mask

    def switch_to(self, team_index: int) -> str | None:
        """Switch active pokemon. Clears volatile state. Returns old pokemon name."""
        if team_index == self.active_index:
            return None
        old = self.active
        old_name = old.name
        # clear volatile state on switch-out
        old.confusion_turns = 0
        old.flinched = False
        old.leech_seeded = False
        old.protected = False
        old.protect_consecutive = 0
        old.recharging = False
        old.charging_move = None
        old.semi_invulnerable = None
        old.locked_move = None
        old.locked_turns = 0
        for stat in old.stat_stages:
            old.stat_stages[stat] = 0
        self.active_index = team_index
        self.active_turns = 1
        return old_name
