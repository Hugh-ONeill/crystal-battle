# resolve_turn(): order actions, execute, handle faints + status effects

from __future__ import annotations

import random as _random

from .actions import Action, Forfeit, Struggle, Switch, UseMove
from .battle_state import RAIN, SANDSTORM, SUN, WEATHER_DURATION, BattleState
from .damage import calc_damage
from .events import (
    ConfusionAppliedEvent,
    ConfusionHitSelfEvent,
    Event,
    FaintEvent,
    FlinchEvent,
    HazeEvent,
    HealEvent,
    LeechSeedAppliedEvent,
    LeechSeedDrainEvent,
    MissEvent,
    MoveEvent,
    PhazeEvent,
    ProtectEvent,
    ResidualDamageEvent,
    ScreenExpiredEvent,
    ScreenSetEvent,
    SpikesDamageEvent,
    SpikesSetEvent,
    StatChangeEvent,
    StatusAppliedEvent,
    StatusCuredEvent,
    StatusMoveEvent,
    StatusPreventedEvent,
    StruggleEvent,
    SwitchEvent,
    WeatherDamageEvent,
    WeatherExpiredEvent,
    WeatherSetEvent,
)
from .move import STRUGGLE
from .player_state import PlayerState
from .stat_stages import MOVE_STAT_EFFECTS, apply_stat_change
from .status import (
    apply_confusion,
    apply_status,
    can_apply_status,
    check_confusion,
    check_move_prevention,
    confusion_from_move,
    effective_speed,
    end_of_turn_damage,
    status_from_move,
)
from .types import TypeChart

# moves handled specially by name
_SPECIAL_MOVES = {
    "Spikes", "Reflect", "Light Screen", "Protect", "Detect",
    "Roar", "Whirlwind", "Leech Seed", "Haze", "Rest",
    "Rapid Spin", "Sunny Day", "Rain Dance", "Sandstorm",
}

# ---- Multi-turn move sets (by move id) ----
_RECHARGE_MOVES = {63}                          # Hyper Beam
_CHARGE_INVULN = {19: "fly", 91: "dig"}         # Fly, Dig (semi-invulnerable)
_CHARGE_EXPOSED = {13, 130, 143}                # Razor Wind, Skull Bash, Sky Attack
_SOLAR_BEAM_ID = 76
_LOCKIN_MOVES = {37, 80, 200}                   # Thrash, Petal Dance, Outrage

_WEATHER_MOVES = {
    "Sunny Day": SUN,
    "Rain Dance": RAIN,
    "Sandstorm": SANDSTORM,
}

# sandstorm immunity
_SANDSTORM_IMMUNE = {"rock", "ground", "steel"}

SCREEN_DURATION = 5


def resolve_turn(
    state: BattleState,
    action1: Action,
    action2: Action,
    type_chart: TypeChart,
) -> list[Event]:
    """
    Resolve one turn of battle. Mutates state, returns events.

    Turn order:
    1. Switches always go first
    2. Higher priority moves go first
    3. Higher speed goes first (paralysis quarters speed)
    4. Coin flip on tie
    """
    state.turn += 1
    events: list[Event] = []

    # clear per-turn volatile state
    state.p1.active.flinched = False
    state.p2.active.flinched = False
    state.p1.active.protected = False
    state.p2.active.protected = False
    state.p1.active.semi_invulnerable = None
    state.p2.active.semi_invulnerable = None

    # ---- Forfeits ----
    if isinstance(action1, Forfeit):
        state.winner = 2
        return events
    if isinstance(action2, Forfeit):
        state.winner = 1
        return events

    # ---- Determine order ----
    order = _determine_order(state, action1, action2)

    for player_num, action in order:
        if state.is_over:
            break

        attacker_state = state.get_player(player_num)
        defender_num = 2 if player_num == 1 else 1
        defender_state = state.get_player(defender_num)

        # skip if attacker fainted (from earlier action this turn)
        if attacker_state.active.is_fainted:
            continue

        new_events = _execute_action(
            state, player_num, attacker_state, defender_state,
            defender_num, action, type_chart,
        )
        events.extend(new_events)

        # residual damage applied after each mon's move (Gen 2 accurate)
        if not state.is_over and not attacker_state.active.is_fainted:
            res_events = _apply_residual_damage(
                player_num, attacker_state, defender_state, defender_num,
            )
            events.extend(res_events)

    # ---- End-of-turn effects (screens, weather) ----
    if not state.is_over:
        eot_events = _end_of_turn(state)
        events.extend(eot_events)

    # check for winner
    state.check_winner()

    # track consecutive turns active (reset happens in switch_to)
    state.p1.active_turns += 1
    state.p2.active_turns += 1

    return events


# ============================================================
# ORDER DETERMINATION
# ============================================================

def _determine_order(
    state: BattleState, action1: Action, action2: Action
) -> list[tuple[int, Action]]:
    """Return [(player_num, action), ...] in execution order."""
    p1_switch = isinstance(action1, Switch)
    p2_switch = isinstance(action2, Switch)

    if p1_switch and not p2_switch:
        return [(1, action1), (2, action2)]
    if p2_switch and not p1_switch:
        return [(2, action2), (1, action1)]
    if p1_switch and p2_switch:
        return _speed_order(state, action1, action2)

    p1_priority = _get_priority(state.p1, action1)
    p2_priority = _get_priority(state.p2, action2)

    if p1_priority > p2_priority:
        return [(1, action1), (2, action2)]
    if p2_priority > p1_priority:
        return [(2, action2), (1, action1)]

    return _speed_order(state, action1, action2)


def _get_priority(player: PlayerState, action: Action) -> int:
    if isinstance(action, UseMove):
        slot = player.active.move_slots[action.slot_index]
        return slot.template.priority
    if isinstance(action, Struggle):
        return 0
    return 0


def _speed_order(
    state: BattleState, action1: Action, action2: Action
) -> list[tuple[int, Action]]:
    speed1 = effective_speed(state.p1.active)
    speed2 = effective_speed(state.p2.active)

    if speed1 > speed2:
        return [(1, action1), (2, action2)]
    if speed2 > speed1:
        return [(2, action2), (1, action1)]
    if state.rng.random() < 0.5:
        return [(1, action1), (2, action2)]
    return [(2, action2), (1, action1)]


# ============================================================
# ACTION EXECUTION
# ============================================================

def _execute_action(
    state: BattleState,
    player_num: int,
    attacker_state: PlayerState,
    defender_state: PlayerState,
    defender_num: int,
    action: Action,
    type_chart: TypeChart,
) -> list[Event]:
    """Execute a single action. Returns events."""
    events: list[Event] = []

    if isinstance(action, Switch):
        old_name = attacker_state.switch_to(action.team_index)
        events.append(SwitchEvent(
            player=player_num,
            from_name=old_name,
            to_name=attacker_state.active.name,
        ))
        # spikes damage on switch-in
        _apply_spikes_damage(attacker_state, player_num, events)
        return events

    if isinstance(action, Struggle):
        move = STRUGGLE
    elif isinstance(action, UseMove):
        slot = attacker_state.active.move_slots[action.slot_index]
        slot.use()
        move = slot.template
    else:
        return events

    attacker = attacker_state.active
    defender = defender_state.active

    # ---- Recharging (Hyper Beam) -- skip turn ----
    if attacker.recharging:
        attacker.recharging = False
        _reset_protect_consecutive(attacker, move)
        return events

    # ---- Locked-in move override (Thrash/Outrage) ----
    if attacker.locked_move is not None:
        move = attacker.locked_move
        attacker.locked_turns -= 1

    # ---- Charging move -- execute on turn 2 ----
    if attacker.charging_move is not None:
        move = attacker.charging_move
        attacker.charging_move = None
        # semi-invulnerable state already cleared at turn start
        # skip straight to damage (no PP cost, no accuracy re-check needed)
        return _execute_damaging_move(
            state, player_num, attacker_state, attacker,
            defender_state, defender, defender_num, move, action, events,
            type_chart,
        )

    # ---- Flinch check ----
    if attacker.flinched:
        events.append(FlinchEvent(player=player_num, pokemon_name=attacker.name))
        _reset_protect_consecutive(attacker, move)
        _break_lock_in(attacker)
        return events

    # ---- Status prevention (sleep/freeze/paralysis) ----
    can_act, reason = check_move_prevention(attacker, state.rng)
    if reason == "woke up":
        events.append(StatusCuredEvent(player=player_num, pokemon_name=attacker.name, status="slp"))
    elif reason == "thawed out":
        events.append(StatusCuredEvent(player=player_num, pokemon_name=attacker.name, status="frz"))

    if not can_act:
        events.append(StatusPreventedEvent(
            player=player_num, pokemon_name=attacker.name,
            status=attacker.status or "", reason=reason or "",
        ))
        _reset_protect_consecutive(attacker, move)
        _break_lock_in(attacker)
        return events

    # ---- Confusion check ----
    hit_self, self_damage = check_confusion(attacker, state.rng)
    if hit_self:
        actual = attacker.take_damage(self_damage)
        events.append(ConfusionHitSelfEvent(
            player=player_num, pokemon_name=attacker.name, damage=actual,
        ))
        if attacker.is_fainted:
            events.append(FaintEvent(player=player_num, pokemon_name=attacker.name))
        _reset_protect_consecutive(attacker, move)
        _break_lock_in(attacker)
        return events

    # ---- Charge turn (Solar Beam, Fly, Dig, Sky Attack, etc.) ----
    mid = move.id
    if attacker.locked_move is None:  # don't re-check for lock-in moves
        if mid == _SOLAR_BEAM_ID and state.weather != SUN:
            attacker.charging_move = move
            return events
        if mid in _CHARGE_INVULN:
            attacker.charging_move = move
            attacker.semi_invulnerable = _CHARGE_INVULN[mid]
            return events
        if mid in _CHARGE_EXPOSED:
            attacker.charging_move = move
            return events

    # ---- Lock-in initiation (Thrash/Outrage) ----
    if mid in _LOCKIN_MOVES and attacker.locked_move is None:
        attacker.locked_move = move
        attacker.locked_turns = state.rng.randint(1, 2)  # 2-3 total turns (this + remaining)

    # ---- Status moves ----
    if move.damage_class == "status":
        applied = _handle_status_move(
            state, player_num, attacker_state, attacker,
            defender_state, defender, defender_num, move, events,
        )
        if not applied:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return events

    # ---- Protect blocks damaging AND status moves targeting opponent ----
    if defender.protected:
        events.append(MissEvent(
            player=player_num, pokemon_name=attacker.name, move_name=move.name,
        ))
        return events

    # ---- Accuracy check (with accuracy/evasion stages) ----
    if move.accuracy is not None:
        effective_acc = _calc_effective_accuracy(move.accuracy, attacker, defender)
        roll = state.rng.randint(1, 100)
        if roll > effective_acc:
            events.append(MissEvent(
                player=player_num,
                pokemon_name=attacker.name,
                move_name=move.name,
            ))
            return events

    return _execute_damaging_move(
        state, player_num, attacker_state, attacker,
        defender_state, defender, defender_num, move, action, events,
        type_chart,
    )


def _execute_damaging_move(
    state, player_num, attacker_state, attacker,
    defender_state, defender, defender_num, move, action, events,
    type_chart=None,
):
    """Execute a damaging move (separated for charge-move reuse)."""

    # ---- Semi-invulnerable dodge (Fly/Dig) ----
    if defender.semi_invulnerable is not None:
        can_hit = False
        if defender.semi_invulnerable == "dig":
            # Earthquake and Magnitude hit underground targets (2x damage in Gen 2)
            if move.name in ("Earthquake", "Magnitude"):
                can_hit = True
        elif defender.semi_invulnerable == "fly":
            # Thunder, Gust, Twister hit airborne targets
            if move.name in ("Thunder", "Gust", "Twister"):
                can_hit = True
        if not can_hit:
            events.append(MissEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
            return events

    # ---- Determine hit count ----
    meta = move.meta or {}
    num_hits = _roll_hit_count(meta, state.rng)

    # ---- Calculate and apply damage (per hit) ----
    total_damage = 0
    any_crit = False
    effectiveness = 1.0

    for hit in range(num_hits):
        if defender.is_fainted:
            break
        damage, effectiveness, is_crit = calc_damage(
            attacker, defender, move, type_chart, state.rng,
            screens=defender_state.side, weather=state.weather,
        )
        actual_hit = defender.take_damage(damage)
        total_damage += actual_hit
        if is_crit:
            any_crit = True

    if isinstance(action, Struggle):
        events.append(StruggleEvent(
            player=player_num,
            pokemon_name=attacker.name,
            damage=total_damage,
        ))
    else:
        events.append(MoveEvent(
            player=player_num,
            pokemon_name=attacker.name,
            move_name=move.name,
            damage=total_damage,
            effectiveness=effectiveness,
            is_crit=any_crit,
            target_hp_remaining=defender.current_hp,
        ))

    # ---- Drain / Recoil ----
    drain_pct = meta.get("drain", 0)
    if drain_pct != 0 and total_damage > 0 and not isinstance(action, Struggle):
        drain_amount = total_damage * abs(drain_pct) // 100
        drain_amount = max(drain_amount, 1)
        if drain_pct > 0:
            actual_heal = attacker.heal(drain_amount)
            if actual_heal > 0:
                events.append(HealEvent(
                    player=player_num, pokemon_name=attacker.name,
                    amount=actual_heal, source="drain",
                ))
        else:
            attacker.take_damage(drain_amount)
            if attacker.is_fainted:
                events.append(FaintEvent(player=player_num, pokemon_name=attacker.name))

    # check for faint
    if defender.is_fainted:
        events.append(FaintEvent(player=defender_num, pokemon_name=defender.name))

    # struggle recoil: 1/4 of damage dealt
    if isinstance(action, Struggle):
        recoil = max(total_damage // 4, 1)
        attacker.take_damage(recoil)
        if attacker.is_fainted:
            events.append(FaintEvent(player=player_num, pokemon_name=attacker.name))

    # ---- Rapid Spin clears hazards and leech seed ----
    if move.name == "Rapid Spin" and total_damage > 0:
        if attacker_state.side.spikes:
            attacker_state.side.spikes = False
        attacker.leech_seeded = False

    # ---- Flinch from damaging move ----
    if not defender.is_fainted and total_damage > 0:
        flinch_chance = meta.get("flinch_chance", 0)
        if flinch_chance > 0:
            roll = state.rng.randint(1, 100)
            if roll <= flinch_chance:
                defender.flinched = True

    # ---- Secondary effects from damaging moves ----
    if not defender.is_fainted and total_damage > 0:
        _try_secondary_effect(state, player_num, attacker, defender, defender_num, move, events)

    # ---- Recharge (Hyper Beam) -- only on hit ----
    if move.id in _RECHARGE_MOVES and total_damage > 0:
        attacker.recharging = True

    # ---- Lock-in end -> confusion ----
    if attacker.locked_move is not None and attacker.locked_turns <= 0:
        attacker.locked_move = None
        attacker.locked_turns = 0
        if apply_confusion(attacker, state.rng):
            events.append(ConfusionAppliedEvent(
                player=player_num, pokemon_name=attacker.name,
            ))

    return events


def _reset_protect_consecutive(attacker, move) -> None:
    """Reset Protect consecutive counter if the mon didn't use Protect/Detect."""
    if move.name not in ("Protect", "Detect"):
        attacker.protect_consecutive = 0


def _break_lock_in(attacker) -> None:
    """Break lock-in state (Thrash/Outrage) on disruption."""
    if attacker.locked_move is not None:
        attacker.locked_move = None
        attacker.locked_turns = 0


def _calc_effective_accuracy(base_accuracy: int, attacker, defender) -> int:
    """Apply accuracy/evasion stages to base accuracy. Gen 2 formula."""
    from .stat_stages import get_stage_multiplier
    acc_stage = attacker.stat_stages.get("accuracy", 0)
    eva_stage = defender.stat_stages.get("evasion", 0)
    net_stage = max(-6, min(6, acc_stage - eva_stage))
    if net_stage == 0:
        return base_accuracy
    num, den = get_stage_multiplier(net_stage)
    return max(1, min(100, base_accuracy * num // den))


def _apply_spikes_damage(player_state: PlayerState, player_num: int, events: list[Event]) -> None:
    """Apply spikes damage on switch-in. Flying types are immune."""
    if not player_state.side.spikes:
        return
    pokemon = player_state.active
    if "flying" in pokemon.types:
        return
    damage = pokemon.max_hp // 8
    actual = pokemon.take_damage(max(damage, 1))
    events.append(SpikesDamageEvent(player=player_num, pokemon_name=pokemon.name, damage=actual))
    if pokemon.is_fainted:
        events.append(FaintEvent(player=player_num, pokemon_name=pokemon.name))


# ============================================================
# MULTI-HIT
# ============================================================

def _roll_hit_count(meta: dict, rng: _random.Random) -> int:
    """Roll number of hits for multi-hit moves. Returns 1 for normal moves.

    Gen 2 distribution for 2-5 range: 2=37.5%, 3=37.5%, 4=12.5%, 5=12.5%
    Fixed-count moves (Double Kick=2, Triple Kick=3) use min_hits directly.
    """
    min_hits = meta.get("min_hits")
    max_hits = meta.get("max_hits")
    if min_hits is None or max_hits is None:
        return 1
    if min_hits == max_hits:
        return min_hits
    roll = rng.randint(1, 8)
    if roll <= 3:
        return 2
    if roll <= 6:
        return 3
    if roll <= 7:
        return 4
    return 5


# ============================================================
# STATUS MOVE HANDLING
# ============================================================

def _handle_status_move(
    state: BattleState,
    player_num: int,
    attacker_state: PlayerState,
    attacker,
    defender_state: PlayerState,
    defender,
    defender_num: int,
    move,
    events: list[Event],
) -> bool:
    """Handle a status-class move. Returns True if something was applied."""
    meta = move.meta or {}
    ailment_id = meta.get("ailment_id", 0)

    # ---- Protect blocks opponent-targeting status moves ----
    # (self-targeting moves like Swords Dance, Rest, Protect itself are not blocked)
    _SELF_TARGET_SPECIALS = {"Protect", "Detect", "Reflect", "Light Screen",
                             "Spikes", "Rest", "Haze", "Sunny Day", "Rain Dance",
                             "Sandstorm", "Rapid Spin"}
    stat_effects = MOVE_STAT_EFFECTS.get(move.name)
    is_self_target = (move.name in _SELF_TARGET_SPECIALS
                      or (stat_effects and all(t == "self" for _, _, t in stat_effects))
                      or (meta.get("healing", 0) > 0 and ailment_id == 0))
    if defender.protected and not is_self_target:
        events.append(MissEvent(
            player=player_num, pokemon_name=attacker.name, move_name=move.name,
        ))
        return True

    # ---- Special moves by name ----
    if move.name in _SPECIAL_MOVES:
        return _handle_special_move(
            state, player_num, attacker_state, attacker,
            defender_state, defender, defender_num, move, events,
        )

    # ---- Stat-change moves (Swords Dance, Growl, etc.) ----
    stat_effects = MOVE_STAT_EFFECTS.get(move.name)
    if stat_effects is not None and ailment_id == 0:
        if move.accuracy is not None:
            roll = state.rng.randint(1, 100)
            if roll > move.accuracy:
                events.append(MissEvent(
                    player=player_num, pokemon_name=attacker.name, move_name=move.name,
                ))
                return True
        # Belly Drum: costs 50% HP, fails if HP <= 50%
        if move.name == "Belly Drum":
            hp_cost = attacker.max_hp // 2
            if attacker.current_hp <= hp_cost:
                events.append(StatusMoveEvent(
                    player=player_num, pokemon_name=attacker.name, move_name=move.name,
                ))
                return True
            attacker.take_damage(hp_cost)
        _apply_stat_effects(stat_effects, player_num, attacker, defender, defender_num, events)
        return True

    # ---- Healing moves (Recover, Soft-Boiled, etc.) ----
    healing_pct = meta.get("healing", 0)
    if healing_pct > 0 and ailment_id == 0:
        heal_amount = attacker.max_hp * healing_pct // 100
        actual_heal = attacker.heal(heal_amount)
        if actual_heal > 0:
            events.append(HealEvent(
                player=player_num, pokemon_name=attacker.name,
                amount=actual_heal, source="recover",
            ))
        else:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return True

    # accuracy check for other status moves
    if move.accuracy is not None:
        roll = state.rng.randint(1, 100)
        if roll > move.accuracy:
            events.append(MissEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
            return True

    # confusion (ailment_id=6)
    if confusion_from_move(ailment_id):
        if apply_confusion(defender, state.rng):
            events.append(ConfusionAppliedEvent(
                player=defender_num, pokemon_name=defender.name,
            ))
        else:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return True

    # non-volatile status (ailment_id 1-5)
    status = status_from_move(move.name, ailment_id)
    if status is None:
        return False

    can, reason = can_apply_status(defender, status)
    if can:
        apply_status(defender, status, state.rng)
        events.append(StatusAppliedEvent(
            player=defender_num, pokemon_name=defender.name, status=status,
        ))
    else:
        events.append(StatusMoveEvent(
            player=player_num, pokemon_name=attacker.name, move_name=move.name,
        ))
    return True


def _handle_special_move(
    state: BattleState,
    player_num: int,
    attacker_state: PlayerState,
    attacker,
    defender_state: PlayerState,
    defender,
    defender_num: int,
    move,
    events: list[Event],
) -> bool:
    """Handle moves with unique mechanics."""

    # ---- Protect / Detect ----
    if move.name in ("Protect", "Detect"):
        # success rate halves each consecutive use: 100%, 50%, 25%, ...
        max_chance = 256
        threshold = max_chance >> attacker.protect_consecutive
        roll = state.rng.randint(1, max_chance)
        if roll <= threshold:
            attacker.protected = True
            attacker.protect_consecutive += 1
            events.append(ProtectEvent(player=player_num, pokemon_name=attacker.name, success=True))
        else:
            attacker.protect_consecutive = 0
            events.append(ProtectEvent(player=player_num, pokemon_name=attacker.name, success=False))
        return True

    # ---- Spikes ----
    if move.name == "Spikes":
        if not defender_state.side.spikes:
            defender_state.side.spikes = True
            events.append(SpikesSetEvent(player=defender_num))
        else:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return True

    # ---- Reflect ----
    if move.name == "Reflect":
        if attacker_state.side.reflect_turns == 0:
            attacker_state.side.reflect_turns = SCREEN_DURATION
            events.append(ScreenSetEvent(player=player_num, screen="reflect"))
        else:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return True

    # ---- Light Screen ----
    if move.name == "Light Screen":
        if attacker_state.side.light_screen_turns == 0:
            attacker_state.side.light_screen_turns = SCREEN_DURATION
            events.append(ScreenSetEvent(player=player_num, screen="light_screen"))
        else:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return True

    # ---- Leech Seed ----
    if move.name == "Leech Seed":
        # accuracy check
        if move.accuracy is not None:
            roll = state.rng.randint(1, 100)
            if roll > move.accuracy:
                events.append(MissEvent(
                    player=player_num, pokemon_name=attacker.name, move_name=move.name,
                ))
                return True
        # grass types immune
        if "grass" in defender.types:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
            return True
        if not defender.leech_seeded:
            defender.leech_seeded = True
            events.append(LeechSeedAppliedEvent(
                player=defender_num, pokemon_name=defender.name,
            ))
        else:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
        return True

    # ---- Roar / Whirlwind (phazing) ----
    if move.name in ("Roar", "Whirlwind"):
        alive_bench = [
            i for i, p in enumerate(defender_state.team)
            if i != defender_state.active_index and not p.is_fainted
        ]
        if not alive_bench:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
            return True
        target_idx = state.rng.choice(alive_bench)
        old_name = defender_state.switch_to(target_idx)
        events.append(PhazeEvent(
            player=defender_num, pokemon_name=old_name or "",
            forced_in=defender_state.active.name,
        ))
        # spikes damage on forced switch-in
        _apply_spikes_damage(defender_state, defender_num, events)
        return True

    # ---- Haze ----
    if move.name == "Haze":
        for ps in (state.p1, state.p2):
            for stat in ps.active.stat_stages:
                ps.active.stat_stages[stat] = 0
        events.append(HazeEvent(player=player_num))
        return True

    # ---- Rest ----
    if move.name == "Rest":
        if attacker.current_hp == attacker.max_hp:
            events.append(StatusMoveEvent(
                player=player_num, pokemon_name=attacker.name, move_name=move.name,
            ))
            return True
        # full heal + self-inflict sleep for 2 turns
        heal_amount = attacker.max_hp - attacker.current_hp
        attacker.heal(heal_amount)
        # clear existing status first
        attacker.clear_status()
        attacker.status = "slp"
        attacker.status_turns = 2
        events.append(HealEvent(
            player=player_num, pokemon_name=attacker.name,
            amount=heal_amount, source="rest",
        ))
        events.append(StatusAppliedEvent(
            player=player_num, pokemon_name=attacker.name, status="slp",
        ))
        return True

    # ---- Weather (Sunny Day, Rain Dance, Sandstorm) ----
    weather = _WEATHER_MOVES.get(move.name)
    if weather is not None:
        state.weather = weather
        state.weather_turns = WEATHER_DURATION
        events.append(WeatherSetEvent(player=player_num, weather=weather))
        return True

    # ---- Rapid Spin (status move portion -- handled as damaging move in main flow) ----
    # Rapid Spin is actually physical/50 power, so it goes through the damaging path
    # This shouldn't be reached, but just in case:
    return False


# ============================================================
# STAT EFFECTS
# ============================================================

def _apply_stat_effects(
    effects: list[tuple[str, int, str]],
    player_num: int,
    attacker, defender, defender_num: int,
    events: list[Event],
) -> None:
    """Apply a list of stat effects and emit events."""
    for stat, stages, target in effects:
        if target == "self":
            actual = apply_stat_change(attacker, stat, stages)
            if actual != 0:
                events.append(StatChangeEvent(
                    player=player_num, pokemon_name=attacker.name,
                    stat=stat, stages=actual,
                ))
        else:
            actual = apply_stat_change(defender, stat, stages)
            if actual != 0:
                events.append(StatChangeEvent(
                    player=defender_num, pokemon_name=defender.name,
                    stat=stat, stages=actual,
                ))


# ============================================================
# SECONDARY EFFECTS
# ============================================================

def _try_secondary_effect(
    state: BattleState,
    player_num: int,
    attacker, defender, defender_num: int,
    move,
    events: list[Event],
) -> None:
    """Roll for secondary effects from a damaging move (status + stat changes)."""
    meta = move.meta
    if meta is None:
        return

    ailment_id = meta.get("ailment_id", 0)
    ailment_chance = meta.get("ailment_chance", 0)

    if ailment_id != 0 and ailment_chance > 0:
        roll = state.rng.randint(1, 100)
        if roll <= ailment_chance:
            if confusion_from_move(ailment_id):
                if apply_confusion(defender, state.rng):
                    events.append(ConfusionAppliedEvent(
                        player=defender_num, pokemon_name=defender.name,
                    ))
            else:
                status = status_from_move(move.name, ailment_id)
                if status is not None:
                    can, _ = can_apply_status(defender, status)
                    if can:
                        apply_status(defender, status, state.rng)
                        events.append(StatusAppliedEvent(
                            player=defender_num, pokemon_name=defender.name, status=status,
                        ))

    stat_chance = meta.get("stat_chance", 0)
    stat_effects = MOVE_STAT_EFFECTS.get(move.name)
    if stat_chance > 0 and stat_effects is not None:
        roll = state.rng.randint(1, 100)
        if roll <= stat_chance:
            _apply_stat_effects(stat_effects, player_num, attacker, defender, defender_num, events)


# ============================================================
# END-OF-TURN
# ============================================================

def _apply_residual_damage(
    player_num: int,
    ps: PlayerState,
    opp_ps: PlayerState,
    opp_num: int,
) -> list[Event]:
    """Apply residual damage to a Pokemon after its move (Gen 2 accurate).

    In Gen 2, residual damage (burn/poison/toxic/leech seed) is applied
    after each Pokemon moves, not in a shared end-of-turn phase.
    """
    events: list[Event] = []
    pokemon = ps.active
    if pokemon.is_fainted:
        return events

    # ---- Residual status damage (burn/poison/toxic) ----
    damage = end_of_turn_damage(pokemon)
    if damage > 0:
        actual = pokemon.take_damage(damage)
        events.append(ResidualDamageEvent(
            player=player_num, pokemon_name=pokemon.name,
            status=pokemon.status or "", damage=actual,
        ))
        if pokemon.is_fainted:
            events.append(FaintEvent(player=player_num, pokemon_name=pokemon.name))
            return events

    # ---- Leech Seed drain ----
    if pokemon.leech_seeded and not pokemon.is_fainted:
        seed_damage = pokemon.max_hp // 8
        actual_drain = pokemon.take_damage(max(seed_damage, 1))
        events.append(LeechSeedDrainEvent(
            player=player_num, pokemon_name=pokemon.name, damage=actual_drain,
        ))
        # heal opponent
        opp_pokemon = opp_ps.active
        if not opp_pokemon.is_fainted:
            opp_pokemon.heal(actual_drain)
        if pokemon.is_fainted:
            events.append(FaintEvent(player=player_num, pokemon_name=pokemon.name))

    return events


def _end_of_turn(state: BattleState) -> list[Event]:
    """Apply end-of-turn effects: screen countdown, sandstorm, weather."""
    events: list[Event] = []

    # ---- Screen countdown ----
    for player_num, ps in [(1, state.p1), (2, state.p2)]:
        if ps.side.reflect_turns > 0:
            ps.side.reflect_turns -= 1
            if ps.side.reflect_turns == 0:
                events.append(ScreenExpiredEvent(player=player_num, screen="reflect"))
        if ps.side.light_screen_turns > 0:
            ps.side.light_screen_turns -= 1
            if ps.side.light_screen_turns == 0:
                events.append(ScreenExpiredEvent(player=player_num, screen="light_screen"))

    # ---- Sandstorm damage ----
    if state.weather == SANDSTORM:
        for player_num, ps in [(1, state.p1), (2, state.p2)]:
            pokemon = ps.active
            if pokemon.is_fainted:
                continue
            if any(t in _SANDSTORM_IMMUNE for t in pokemon.types):
                continue
            damage = max(pokemon.max_hp // 8, 1)
            actual = pokemon.take_damage(damage)
            events.append(WeatherDamageEvent(
                player=player_num, pokemon_name=pokemon.name, damage=actual,
            ))
            if pokemon.is_fainted:
                events.append(FaintEvent(player=player_num, pokemon_name=pokemon.name))

    # ---- Leftovers healing ----
    for player_num, ps in [(1, state.p1), (2, state.p2)]:
        pokemon = ps.active
        if pokemon.is_fainted:
            continue
        if pokemon.item == "leftovers" and pokemon.current_hp < pokemon.max_hp:
            heal_amt = max(pokemon.max_hp // 16, 1)
            actual = pokemon.heal(heal_amt)
            if actual > 0:
                events.append(HealEvent(
                    player=player_num, pokemon_name=pokemon.name, amount=actual,
                ))

    # ---- Berry items (consume on use) ----
    for player_num, ps in [(1, state.p1), (2, state.p2)]:
        pokemon = ps.active
        if pokemon.is_fainted:
            continue
        if pokemon.item == "miracleberry" and pokemon.status is not None:
            pokemon.clear_status()
            pokemon.item = None
            events.append(StatusCuredEvent(
                player=player_num, pokemon_name=pokemon.name, status="cured",
            ))
        elif pokemon.item == "mintberry" and pokemon.status == "slp":
            pokemon.clear_status()
            pokemon.item = None
            events.append(StatusCuredEvent(
                player=player_num, pokemon_name=pokemon.name, status="slp",
            ))

    # ---- Weather countdown ----
    if state.weather is not None:
        state.weather_turns -= 1
        if state.weather_turns <= 0:
            events.append(WeatherExpiredEvent(weather=state.weather))
            state.weather = None
            state.weather_turns = 0

    return events


# ============================================================
# FORCED SWITCHES
# ============================================================

def resolve_forced_switches(
    state: BattleState,
    switch1: Switch | None,
    switch2: Switch | None,
) -> list[Event]:
    """Handle forced switches after faints (mid-turn or end-of-turn)."""
    events: list[Event] = []

    for player_num, switch in [(1, switch1), (2, switch2)]:
        if switch is None:
            continue
        ps = state.get_player(player_num)
        if ps.must_switch:
            old_name = ps.switch_to(switch.team_index)
            events.append(SwitchEvent(
                player=player_num,
                from_name=old_name,
                to_name=ps.active.name,
            ))
            # spikes damage on forced switch-in
            _apply_spikes_damage(ps, player_num, events)

    state.check_winner()
    return events
