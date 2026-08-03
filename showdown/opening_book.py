"""Per-team opening book: condition-gated canonical startup sequences.

WHY SHADOW FIRST. The book is a lever over the search, and every lever this
campaign has aimed at the search has needed its fire rate measured before its
strength. The move-net lost 51pp by injecting preferences everywhere; the
overlay earned its keep only because shadow mode showed what it would have
changed before it changed anything. So this module's FIRST job is not to play
moves — it is to answer three questions on real games:

  1. On what fraction of GAMES would the book change at least one decision?
     Per-TURN fire rate is the wrong instrument and would badly undersell
     this: an opening book is supposed to fire a handful of times and then
     go quiet, and an opening decision is not worth its frequency. The lead
     is one decision per game and it gates the whole plan — the sun teams
     led their own setter 8% of the time while running a mon that REQUIRES
     sun, and troom1/troom2 sat 23pp apart on whether the setup line got
     executed at the start. Per-game exposure is the number that matters.
  2. When it fires, does it AGREE with MCTS? Agreement means the search
     already knows the line and the book is dead weight.
  3. When it disagrees, what is the MARGIN — the visit-share gap between the
     book's move and the search's pick? That is the number that decides which
     strength is even viable: a root prior can flip a near-tie for free, but
     it cannot overturn a 0.6-share conviction, and if it could we would not
     want it to (that is the catastrophe veto doing its job).

Answering those costs nothing: shadow logging never touches the returned
order. Only after they are answered does `weighted` (root prior) or
`scripted` (pre-search override) get a paired A/B, on the book's teams only.

Condition vocabulary is deliberately tiny and every key is checkable against
a poke-env Battle — see opening_book.json's SCHEMA block. A step whose
condition cannot be evaluated is simply skipped, never guessed at.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_BOOK_PATH = Path(__file__).parent / "opening_book.json"

# how the book's field vocabulary maps onto what poke-env exposes
_WEATHERS = {"snow": "SNOW", "hail": "HAIL", "sun": "SUNNYDAY",
             "rain": "RAINDANCE", "sand": "SANDSTORM"}
_TERRAINS = {"electricterrain": "ELECTRIC_TERRAIN",
             "grassyterrain": "GRASSY_TERRAIN",
             "psychicterrain": "PSYCHIC_TERRAIN",
             "mistyterrain": "MISTY_TERRAIN"}
_SCREENS = ("REFLECT", "LIGHT_SCREEN", "AURORA_VEIL")


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class OpeningBook:
    """Reads opening_book.json and reports what it WOULD have played."""

    def __init__(self, path=None, mode: str = "shadow"):
        with open(path or _BOOK_PATH) as f:
            self.books = json.load(f)["books"]
        self.mode = mode
        # Identify by FULL ROSTER read from the paste on disk, not by name:
        # the bench swaps a per-lane team file, so at runtime the original
        # filename is gone. One source of truth — the same pastes the
        # validator cross-checks the steps against. Falls back to the species
        # the steps mention if a paste is missing, which is weaker (2-4 mons
        # can collide across teams) and so is only ever a fallback.
        self._roster_index = {}
        for key in self.books:
            self._roster_index[key] = self._paste_roster(key) or frozenset(
                _norm(s) for s in self._species_of(key))
        self.log: list[dict] = []
        self._used: set[str] = set()
        self._team_key: str | None = None

    # ---- team identification ------------------------------------------

    @staticmethod
    def _paste_roster(key: str) -> frozenset:
        """The team's six species, from the paste the validator also reads."""
        import glob
        hits = glob.glob(str(Path(__file__).parent / "teams" / "*" / f"{key}.txt"))
        if not hits:
            return frozenset()
        try:
            from showdown.local_battle import parse_showdown_team
            return frozenset(_norm(m["species"]) for m in
                             parse_showdown_team(Path(hits[0]).read_text()))
        except Exception:
            return frozenset()

    def _species_of(self, key: str) -> list[str]:
        """Species the book's steps reference, used only as a fallback when
        the paste name does not match."""
        book = self.books[key]
        out = [book.get("lead", "")]
        for step in book.get("steps", []):
            when = step.get("when", {})
            if "active" in when:
                out.append(when["active"])
            do = str(step.get("do", ""))
            if do.startswith("switch "):
                out.append(do[7:])
        return [s for s in out if s]

    def identify(self, team_name: str | None, roster) -> str | None:
        """Match this game's team to a book entry. Paste name first — it is
        exact and the bench always has it — then roster containment, so a
        renamed or re-hashed copy of the same team still books."""
        if team_name:
            base = re.sub(r"^G\d+_", "", str(team_name)).removesuffix(".txt")
            for key in self.books:
                if base == key or base.startswith(key):
                    self._team_key = key
                    return key
        ours = {_norm(s) for s in roster or ()}
        if ours:
            for key, need in self._roster_index.items():
                if need and need <= ours:
                    self._team_key = key
                    return key
        self._team_key = None
        return None

    # ---- condition evaluation -----------------------------------------

    @staticmethod
    def _fields_up(battle) -> set[str]:
        up = set()
        for w in (battle.weather or {}):
            for name, enum in _WEATHERS.items():
                if getattr(w, "name", "") == enum:
                    up.add(name)
        for f in (battle.fields or {}):
            fname = getattr(f, "name", "")
            for name, enum in _TERRAINS.items():
                if fname == enum:
                    up.add(name)
            if fname == "TRICK_ROOM":
                up.add("trickroom")
        for c in (battle.side_conditions or {}):
            if getattr(c, "name", "") in _SCREENS:
                up.add("screens")
        return up

    def _holds(self, when: dict, battle) -> bool:
        active = battle.active_pokemon
        if "active" in when:
            if active is None or _norm(active.species) != _norm(when["active"]):
                return False
        if "max_turn" in when and battle.turn > int(when["max_turn"]):
            return False
        if "hp_at_least" in when:
            if active is None:
                return False
            frac = getattr(active, "current_hp_fraction", None)
            if frac is None or frac < float(when["hp_at_least"]):
                return False
        up = None
        if "field_up" in when:
            up = self._fields_up(battle)
            if when["field_up"] not in up:
                return False
        if "field_down" in when:
            up = self._fields_up(battle) if up is None else up
            if when["field_down"] in up:
                return False
        if "unused" in when and _norm(when["unused"]) in self._used:
            return False
        return True

    def suggest(self, battle) -> dict | None:
        """First step whose condition holds. No match = the search decides,
        which is the default for every turn this file says nothing about."""
        if self._team_key is None:
            return None
        for step in self.books[self._team_key].get("steps", []):
            if self._holds(step.get("when", {}), battle):
                return step
        return None

    def lead(self) -> str | None:
        if self._team_key is None:
            return None
        return self.books[self._team_key].get("lead")

    # ---- shadow instrumentation ---------------------------------------

    def observe_lead(self, chosen) -> dict | None:
        """The once-per-game decision, logged separately from the steps.

        Kept distinct because it is not comparable to them: there is no
        visit-share margin at preview (the pool is a maximin over 6x6
        pairings, and the pick among near-ties is a uniform draw), so the
        only thing to record is whether the book's lead is the one that
        actually went out. That single bit is the book's largest lever —
        every later step is conditioned on the right mon being there.
        """
        want = self.lead()
        if want is None:
            return None
        entry = {
            "team": self._team_key, "turn": 0, "kind": "lead",
            "book": want, "mcts": _norm(chosen) if chosen else None,
            "agree": chosen is not None and _norm(chosen) == _norm(want),
        }
        self.log.append(entry)
        return entry

    def note_used(self, played: str):
        """Record a move we actually played, so `unused` steps retire.

        Takes the player's display label, which may be "earthquake (tera)" or
        "switch Torkoal" — normalising blindly would turn the first into
        `earthquaketera` and the step would never retire.
        """
        if not played:
            return
        label = str(played).split(" (")[0]
        if label.startswith("switch "):
            return
        self._used.add(_norm(label))

    def reset(self):
        self._used.clear()
        self._team_key = None

    def observe(self, battle, ranked) -> dict | None:
        """Record what the book WOULD have played against what MCTS picked.

        Never mutates `ranked`. `margin` is the visit-share gap the book would
        have to overcome; `book_rank` is where the search already had it.
        """
        step = self.suggest(battle)
        if step is None or not ranked:
            return None
        want = _norm(str(step["do"]).replace("switch ", "switch|"))
        top = ranked[0]
        total = sum(max(0, getattr(r, "visits", 0)) for r in ranked) or 1

        def share(r):
            return max(0, getattr(r, "visits", 0)) / total

        found = None
        for i, r in enumerate(ranked):
            rc = _norm(str(r.move_choice).replace("switch ", "switch|"))
            if rc == want or rc.removesuffix("tera") == want:
                found = (i, r)
                break
        entry = {
            "team": self._team_key,
            "turn": battle.turn,
            "book": step["do"],
            "mcts": top.move_choice,
            "agree": found is not None and found[0] == 0,
            "book_rank": found[0] if found else None,
            "book_share": round(share(found[1]), 4) if found else None,
            "top_share": round(share(top), 4),
            "legal": found is not None,
        }
        if found is not None:
            entry["margin"] = round(share(top) - share(found[1]), 4)
        self.log.append(entry)
        return entry

    def dump(self, path):
        if not self.log:
            return
        with open(path, "a") as f:
            for row in self.log:
                f.write(json.dumps(row) + "\n")
        self.log.clear()


def book_mode() -> str:
    """off | shadow | weighted | scripted. Shadow is instrumentation only."""
    return (os.environ.get("CB_OPENING_BOOK") or "off").strip().lower()
