#!/usr/bin/env python3
# Format and analyze Crystal Battle replay JSON files
# Usage: python tools/replay.py replays/run33_750k/
#        python tools/replay.py replays/run33_750k/game_000.json
#        python tools/replay.py replays/run33_750k/ --no-analysis

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.types import TypeChart

_TC = TypeChart.load()

# ============================================================
# TYPE HELPERS
# ============================================================

def _eff(move_type: str, target_types: list[str]) -> float:
    return _TC.combined_effectiveness(move_type, target_types)


def _eff_label(eff: float) -> str:
    if eff == 0:
        return "IMMUNE"
    elif eff >= 4:
        return "4x SE"
    elif eff >= 2:
        return "SE"
    elif eff <= 0.25:
        return "4x resist"
    elif eff <= 0.5:
        return "resist"
    return ""


def _p2_move_used(turn: dict) -> dict | None:
    """Infer P2's move from PP changes between before/after."""
    before = turn["p2_active"]["moves"]
    after = turn["p2_active_after"]["moves"]
    if turn["p2_active_after"]["name"] != turn["p2_active"]["name"]:
        return None  # p2 switched or fainted, can't infer
    for b, a in zip(before, after):
        b_pp = int(b["pp"].split("/")[0])
        a_pp = int(a["pp"].split("/")[0])
        if a_pp < b_pp:
            return a
    return None


# ============================================================
# FORMATTER
# ============================================================

def format_game(data: dict) -> str:
    lines: list[str] = []
    winner = data["winner"]
    result = "P1 WIN" if winner == "p1" else "P2 WIN"
    p1_left = data["p1_remaining"]
    p2_left = data["p2_remaining"]
    total = data["total_turns"]

    lines.append(f"{'=' * 60}")
    lines.append(f"Game {data['game']}: {result} ({p1_left}v{p2_left}, {total} turns) vs {data['baseline']}")
    lines.append(f"{'=' * 60}")

    # team rosters
    p1_names = [p["name"] for p in data["p1_team"]]
    p2_names = [p["name"] for p in data["p2_team"]]
    p1_types = {p["name"]: p["types"] for p in data["p1_team"]}
    p2_types = {p["name"]: p["types"] for p in data["p2_team"]}
    all_types = {**p1_types, **p2_types}

    lines.append(f"P1: {' | '.join(p1_names)}")
    lines.append(f"P2: {' | '.join(p2_names)}")
    lines.append("")

    prev_p1_fainted = False

    for i, turn in enumerate(data["turns"]):
        tn = turn["turn"]
        p1_before = turn["p1_active"]
        p2_before = turn["p2_active"]
        p1_after = turn["p1_active_after"]
        p2_after = turn["p2_active_after"]
        action = turn["p1_action"]

        p1_hp_before = p1_before["hp_pct"]
        p1_hp_after = p1_after["hp_pct"]
        p2_hp_before = p2_before["hp_pct"]
        p2_hp_after = p2_after["hp_pct"]

        # detect if P2 switched (different mon after)
        p2_switched = p2_after["name"] != p2_before["name"] and p2_after["hp_pct"] > 0

        # P1 action
        if action["type"] == "move":
            move_name = action["name"]
            move_type = action["move_type"]
            target_types = all_types.get(p2_before["name"], [])
            # self-targeting moves shouldn't show effectiveness vs opponent
            _SELF_TARGET = {"Rest", "Soft-Boiled", "Recover", "Milk Drink", "Moonlight",
                            "Morning Sun", "Synthesis", "Swords Dance", "Curse", "Agility",
                            "Amnesia", "Growth", "Meditate", "Belly Drum", "Dragon Dance",
                            "Calm Mind", "Bulk Up", "Rain Dance", "Sunny Day", "Sandstorm",
                            "Reflect", "Light Screen", "Safeguard", "Spikes", "Substitute",
                            "Protect", "Detect", "Endure", "Sleep Talk"}
            if move_name in _SELF_TARGET:
                eff_tag = ""
            else:
                eff = _eff(move_type, target_types) if target_types else 1.0
                eff_str = _eff_label(eff)
                eff_tag = f" [{eff_str}]" if eff_str else ""

            sw_tag = " [fsw]" if prev_p1_fainted else ""
            p1_col = f"{p1_before['name']}{sw_tag}({p1_hp_before:.0f}%)"

            if p2_after["hp_pct"] <= 0 and p2_after["name"] == p2_before["name"]:
                result_str = f"-> {p2_before['name']} KO"
            elif p2_after["name"] != p2_before["name"]:
                # opp fainted and new mon came in, or opp switched
                result_str = f"-> {p2_before['name']} KO"
            else:
                dmg = p2_hp_before - p2_hp_after
                if dmg > 0.5:
                    result_str = f"-> {p2_before['name']}({p2_hp_after:.0f}%)"
                elif action["power"] == 0:
                    result_str = f"-> {p2_before['name']}"
                else:
                    result_str = f"-> {p2_before['name']} (no dmg)"

            # P1 damage taken
            p1_dmg = p1_hp_before - p1_hp_after
            if p1_after["hp_pct"] <= 0:
                taken_str = "| FAINTED"
            elif p1_dmg > 0.5:
                taken_str = f"| took {p1_dmg:.0f}%({p1_hp_after:.0f}%)"
            elif p1_dmg < -0.5:
                taken_str = f"| healed to {p1_hp_after:.0f}%"
            else:
                taken_str = ""

            # P2 move inferred
            p2_move = _p2_move_used(turn)
            p2_str = ""
            if p2_move and p2_move["name"]:
                p2_str = f"  P2: {p2_move['name']}"

            lines.append(
                f"T{tn:02d}  {p1_col:<22s} {move_name:<16s}{eff_tag:<12s} "
                f"{result_str:<28s} {taken_str}{p2_str}"
            )

        elif action["type"] == "switch":
            sw_type = "fsw" if prev_p1_fainted else "sw"
            to_name = action["to"]
            to_hp = action["to_hp_pct"]

            # damage taken on switch-in
            p1_dmg = to_hp - p1_hp_after if p1_after["name"] == to_name else 0
            if p1_after["hp_pct"] <= 0:
                taken_str = "| FAINTED on switch-in"
            elif p1_dmg > 0.5:
                taken_str = f"| took {p1_dmg:.0f}%({p1_hp_after:.0f}%)"
            else:
                taken_str = ""

            p2_move = _p2_move_used(turn)
            p2_str = ""
            if p2_move and p2_move["name"]:
                p2_str = f"  P2: {p2_move['name']}"

            lines.append(
                f"T{tn:02d}  [{sw_type} {to_name}({to_hp:.0f}%)]{'':.<30s} "
                f"vs {p2_before['name']}({p2_hp_before:.0f}%) {'':.<13s} {taken_str}{p2_str}"
            )

        # track faint for next turn's forced switch detection
        prev_p1_fainted = p1_after["hp_pct"] <= 0

        # note P2 switch-in
        if p2_switched and p2_after["name"] != p2_before["name"]:
            lines.append(f"      P2 sends in {p2_after['name']}({p2_after['hp_pct']:.0f}%)")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# ANALYZER
# ============================================================

# status moves that can't affect certain types
_STATUS_IMMUNITIES = {
    "poison": {"steel", "poison"},
    "toxic": {"steel", "poison"},
    "paralysis": {"electric"},
    "burn": {"fire"},
    "freeze": {"ice"},
}

# ailment names from move meta
_AILMENT_MOVES = {
    "Toxic": "toxic",
    "Poison Powder": "poison",
    "Stun Spore": "paralysis",
    "Thunder Wave": "paralysis",
    "Glare": "paralysis",
    "Hypnosis": "sleep",
    "Sleep Powder": "sleep",
    "Lovely Kiss": "sleep",
    "Spore": "sleep",
    "Will-O-Wisp": "burn",
    "Confuse Ray": "confusion",
}

_ONE_TIME_MOVES = {"Spikes", "Light Screen", "Reflect", "Safeguard"}

_HEALING_MOVES = {"Rest", "Soft-Boiled", "Recover", "Milk Drink", "Moonlight",
                  "Morning Sun", "Synthesis", "Sleep Talk"}

_SLEEP_MOVES = {"Hypnosis", "Sleep Powder", "Lovely Kiss", "Spore", "Sing"}
_RECOVERY_MOVES = {"Soft-Boiled", "Recover", "Milk Drink", "Moonlight",
                   "Morning Sun", "Synthesis", "Rest"}


def analyze_game(data: dict) -> tuple[list[str], list[str]]:
    flags: list[str] = []
    turns = data["turns"]
    p1_types = {p["name"]: p["types"] for p in data["p1_team"]}
    p2_types = {p["name"]: p["types"] for p in data["p2_team"]}
    all_types = {**p1_types, **p2_types}

    # tracking
    prev_move: str | None = None
    move_streak = 0
    prev_switches: list[int] = []
    spikes_up = False
    screens_up: set[str] = set()
    p1_alive = {p["name"] for p in data["p1_team"]}

    for i, turn in enumerate(turns):
        tn = turn["turn"]
        action = turn["p1_action"]
        p1_before = turn["p1_active"]
        p2_before = turn["p2_active"]
        p1_after = turn["p1_active_after"]
        p2_after = turn["p2_active_after"]
        target_types = all_types.get(p2_before["name"], [])

        # ---- Type-immune move ----
        if action["type"] == "move" and action["power"] > 0:
            eff = _eff(action["move_type"], target_types)
            if eff == 0:
                flags.append(
                    f"  T{tn:02d} TYPE IMMUNE: {p1_before['name']} used {action['name']} "
                    f"({action['move_type']}) vs {p2_before['name']} ({'/'.join(target_types)})"
                )

        # ---- SE available but not used ----
        # moves that require special conditions and shouldn't be flagged
        _CONDITIONAL_MOVES = {"Dream Eater", "Snore", "Sleep Talk"}

        if action["type"] == "move" and action["power"] > 0:
            used_eff = _eff(action["move_type"], target_types)
            best_eff = used_eff
            best_move = None
            for m in p1_before["moves"]:
                if m["power"] > 0 and m["name"] not in _CONDITIONAL_MOVES:
                    e = _eff(m["type"], target_types)
                    if e > best_eff:
                        best_eff = e
                        best_move = m
            if best_move and best_eff >= 2.0 and used_eff < best_eff:
                flags.append(
                    f"  T{tn:02d} MISSED SE: {p1_before['name']} used {action['name']} "
                    f"({_eff_label(used_eff) or 'neutral'}) but had {best_move['name']} "
                    f"({_eff_label(best_eff)}) vs {p2_before['name']}"
                )

        # ---- Status immunity ----
        if action["type"] == "move" and action["power"] == 0:
            move_name = action["name"]
            if move_name in _AILMENT_MOVES:
                ailment = _AILMENT_MOVES[move_name]
                if ailment in _STATUS_IMMUNITIES:
                    immune_types = _STATUS_IMMUNITIES[ailment]
                    if immune_types & set(target_types):
                        flags.append(
                            f"  T{tn:02d} STATUS IMMUNE: {p1_before['name']} used {move_name} "
                            f"vs {p2_before['name']} ({'/'.join(target_types)}) -- immune to {ailment}"
                        )

        # ---- One-time move when already active ----
        if action["type"] == "move" and action["name"] in _ONE_TIME_MOVES:
            move_name = action["name"]
            if move_name == "Spikes" and spikes_up:
                flags.append(
                    f"  T{tn:02d} WASTED: {p1_before['name']} used Spikes but spikes already up"
                )
            elif move_name in ("Light Screen", "Reflect", "Safeguard") and move_name in screens_up:
                flags.append(
                    f"  T{tn:02d} WASTED: {p1_before['name']} used {move_name} but already active"
                )

        # track spikes/screens state
        if action["type"] == "move":
            if action["name"] == "Spikes":
                spikes_up = True
            elif action["name"] in ("Light Screen", "Reflect", "Safeguard"):
                screens_up.add(action["name"])

        # ---- Move locking (same move 3+ times in a row) ----
        if action["type"] == "move":
            if action["name"] == prev_move:
                move_streak += 1
            else:
                if move_streak >= 3:
                    flags.append(
                        f"  T{tn - move_streak:02d}-T{tn - 1:02d} MOVE LOCK: {prev_move} x{move_streak} in a row"
                    )
                prev_move = action["name"]
                move_streak = 1
        else:
            if move_streak >= 3:
                flags.append(
                    f"  T{tn - move_streak:02d}-T{tn - 1:02d} MOVE LOCK: {prev_move} x{move_streak} in a row"
                )
            prev_move = None
            move_streak = 0

        # ---- Status into death (excludes healing moves) ----
        if (action["type"] == "move" and action["power"] == 0
                and p1_after["hp_pct"] <= 0
                and action["name"] not in _HEALING_MOVES):
            has_damage_move = any(m["power"] > 0 for m in p1_before["moves"]
                                  if int(m["pp"].split("/")[0]) > 0)
            if has_damage_move:
                flags.append(
                    f"  T{tn:02d} STATUS->DEATH: {p1_before['name']}({p1_before['hp_pct']:.0f}%) "
                    f"used {action['name']} (status) and fainted -- had damage moves available"
                )

        # ---- Switch-looping (2+ consecutive voluntary switches) ----
        is_forced = (i > 0 and turns[i - 1]["p1_active_after"]["hp_pct"] <= 0)
        if action["type"] == "switch" and not is_forced:
            prev_switches.append(tn)
        else:
            if len(prev_switches) >= 2:
                flags.append(
                    f"  T{prev_switches[0]:02d}-T{prev_switches[-1]:02d} SWITCH LOOP: "
                    f"{len(prev_switches)} consecutive voluntary switches"
                )
            prev_switches = []

        # ---- Suicidal switch-in ----
        if action["type"] == "switch":
            to_name = action["to"]
            to_types = all_types.get(to_name, [])
            if to_types:
                # worst effectiveness from opp moves against switch target
                def _worst_opp_eff(mon_types):
                    worst = 0.0
                    for m in p2_before["moves"]:
                        if m["power"] > 0:
                            worst = max(worst, _eff(m["type"], mon_types))
                    return worst

                worst_eff = _worst_opp_eff(to_types)
                if worst_eff >= 4.0:
                    worst_move = max((m for m in p2_before["moves"] if m["power"] > 0),
                                     key=lambda m: _eff(m["type"], to_types))
                    flags.append(
                        f"  T{tn:02d} BAD SWITCH: {to_name} ({'/'.join(to_types)}) "
                        f"into {p2_before['name']} who has {worst_move['name']} ({_eff_label(worst_eff)})"
                    )

                # check if a better bench option existed
                if worst_eff >= 2.0 and not is_forced:
                    active_name = p1_before["name"]
                    bench = p1_alive - {active_name, to_name}
                    best_alt = None
                    best_alt_eff = worst_eff
                    for mon in bench:
                        mon_types = all_types.get(mon, [])
                        if mon_types:
                            alt_eff = _worst_opp_eff(mon_types)
                            if alt_eff < best_alt_eff:
                                best_alt_eff = alt_eff
                                best_alt = mon
                    if best_alt and best_alt_eff < worst_eff:
                        flags.append(
                            f"  T{tn:02d} BETTER SWITCH: {best_alt} "
                            f"(worst {_eff_label(best_alt_eff) or 'neutral'}) "
                            f"was safer than {to_name} "
                            f"(worst {_eff_label(worst_eff)})"
                        )

        # track faints
        if p1_after["hp_pct"] <= 0:
            p1_alive.discard(p1_before["name"])

    # ============================================================
    # STRATEGIC PLAY DETECTION
    # ============================================================
    strats: list[str] = []

    # ---- Sleep + follow-up ----
    for i, turn in enumerate(turns):
        action = turn["p1_action"]
        if action["type"] == "move" and action["name"] in _SLEEP_MOVES:
            p2_before = turn["p2_active"]
            # check if next turn is an attack on the same target
            if i + 1 < len(turns):
                nxt = turns[i + 1]
                nxt_action = nxt["p1_action"]
                if (nxt_action["type"] == "move" and nxt_action["power"] > 0
                        and nxt["p2_active"]["name"] == p2_before["name"]):
                    strats.append(
                        f"  T{turn['turn']:02d}-T{nxt['turn']:02d} SLEEP COMBO: "
                        f"{action['name']} -> {nxt_action['name']} vs {p2_before['name']}"
                    )

    # ---- Toxic stall (Toxic then recovery within 3 turns) ----
    for i, turn in enumerate(turns):
        action = turn["p1_action"]
        if action["type"] == "move" and action["name"] == "Toxic":
            target = turn["p2_active"]["name"]
            for j in range(i + 1, min(i + 4, len(turns))):
                nxt = turns[j]
                if (nxt["p1_action"]["type"] == "move"
                        and nxt["p1_action"]["name"] in _RECOVERY_MOVES
                        and nxt["p2_active"]["name"] == target):
                    strats.append(
                        f"  T{turn['turn']:02d}-T{nxt['turn']:02d} TOXIC STALL: "
                        f"Toxic + {nxt['p1_action']['name']} vs {target}"
                    )
                    break

    # ---- SE sweep (3+ consecutive SE/4xSE attacks) ----
    se_streak = 0
    se_start = 0
    for i, turn in enumerate(turns):
        action = turn["p1_action"]
        if action["type"] == "move" and action["power"] > 0:
            target_types = all_types.get(turn["p2_active"]["name"], [])
            eff = _eff(action["move_type"], target_types) if target_types else 1.0
            if eff >= 2.0:
                if se_streak == 0:
                    se_start = turn["turn"]
                se_streak += 1
            else:
                if se_streak >= 3:
                    strats.append(
                        f"  T{se_start:02d}-T{turns[i-1]['turn']:02d} SE SWEEP: "
                        f"{se_streak} consecutive super-effective hits"
                    )
                se_streak = 0
        else:
            if se_streak >= 3:
                strats.append(
                    f"  T{se_start:02d}-T{turns[i-1]['turn']:02d} SE SWEEP: "
                    f"{se_streak} consecutive super-effective hits"
                )
            se_streak = 0
    if se_streak >= 3:
        last_tn = turns[-1]["turn"]
        strats.append(
            f"  T{se_start:02d}-T{last_tn:02d} SE SWEEP: "
            f"{se_streak} consecutive super-effective hits"
        )

    # ---- Smart switch (switch into mon that resists opp's last move) ----
    for i, turn in enumerate(turns):
        action = turn["p1_action"]
        if action["type"] == "switch" and i > 0:
            # was previous turn's P2 move a damaging one?
            prev_p2_move = _p2_move_used(turns[i - 1])
            if prev_p2_move and prev_p2_move["power"] > 0:
                to_types = all_types.get(action["to"], [])
                if to_types:
                    eff = _eff(prev_p2_move["type"], to_types)
                    if eff <= 0.5:
                        strats.append(
                            f"  T{turn['turn']:02d} SMART SWITCH: {action['to']} "
                            f"resists {prev_p2_move['name']} ({_eff_label(eff)})"
                        )

    # flush trailing streaks
    if move_streak >= 3:
        last_tn = turns[-1]["turn"]
        flags.append(
            f"  T{last_tn - move_streak + 1:02d}-T{last_tn:02d} MOVE LOCK: {prev_move} x{move_streak} in a row"
        )
    if len(prev_switches) >= 2:
        flags.append(
            f"  T{prev_switches[0]:02d}-T{prev_switches[-1]:02d} SWITCH LOOP: "
            f"{len(prev_switches)} consecutive voluntary switches"
        )

    return flags, strats


# ============================================================
# MAIN
# ============================================================

def process_file(path: Path, show_analysis: bool = True) -> str:
    data = json.load(open(path))
    output = format_game(data)
    if show_analysis:
        flags, strats = analyze_game(data)
        if flags:
            output += "ISSUES:\n" + "\n".join(flags) + "\n"
        else:
            output += "ISSUES: none detected\n"
        if strats:
            output += "STRATEGY:\n" + "\n".join(strats) + "\n"
    return output


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <replay_dir_or_file> [--no-analysis]")
        sys.exit(1)

    target = Path(sys.argv[1])
    show_analysis = "--no-analysis" not in sys.argv

    if target.is_file():
        print(process_file(target, show_analysis))
    elif target.is_dir():
        files = sorted(target.glob("game_*.json"))
        if not files:
            print(f"No game_*.json files found in {target}")
            sys.exit(1)
        for f in files:
            print(process_file(f, show_analysis))
    else:
        print(f"Not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
