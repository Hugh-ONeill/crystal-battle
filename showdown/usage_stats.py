# GSC OU usage statistics from Smogon (Oct 2025, 6615 battles)
# source: https://www.smogon.com/stats/2025-10/moveset/gen2ou-0.txt
#
# used for opponent team prediction when we have partial information

# (species_name, usage%, top_moves_with_%, top_items_with_%, top_teammates_with_%)
# moves/items/teammates are (name, probability) tuples

USAGE_STATS = {
    "snorlax": {
        "usage": 82.09,
        "moves": [("rest", 0.646), ("earthquake", 0.604), ("doubleedge", 0.584),
                  ("curse", 0.560), ("sleeptalk", 0.40), ("bodyslam", 0.20),
                  ("selfdestruct", 0.15), ("fireblast", 0.10), ("thunder", 0.08)],
        "items": [("leftovers", 0.903), ("miracleberry", 0.065)],
        "teammates": [("zapdos", 0.507), ("cloyster", 0.471), ("gengar", 0.316)],
    },
    "zapdos": {
        "usage": 45.11,
        "moves": [("hiddenpowerice", 0.662), ("thunder", 0.607), ("rest", 0.511),
                  ("sleeptalk", 0.480), ("thunderwave", 0.15), ("whirlwind", 0.10),
                  ("thunderbolt", 0.08)],
        "items": [("leftovers", 0.969)],
        "teammates": [("snorlax", 0.927), ("cloyster", 0.621), ("gengar", 0.406)],
    },
    "cloyster": {
        "usage": 43.27,
        "moves": [("spikes", 0.965), ("explosion", 0.903), ("surf", 0.882),
                  ("toxic", 0.761), ("icebeam", 0.05)],
        "items": [("leftovers", 0.928), ("miracleberry", 0.054)],
        "teammates": [("snorlax", 0.891), ("zapdos", 0.642), ("gengar", 0.387)],
    },
    "gengar": {
        "usage": 32.12,
        "moves": [("explosion", 0.771), ("icepunch", 0.759), ("thunderbolt", 0.596),
                  ("destinybond", 0.309), ("dynamicpunch", 0.25), ("hypnosis", 0.15),
                  ("thunder", 0.10), ("thief", 0.08)],
        "items": [("leftovers", 0.636), ("miracleberry", 0.191)],
        "teammates": [("snorlax", 0.805), ("zapdos", 0.566), ("cloyster", 0.521)],
    },
    "exeggutor": {
        "usage": 31.17,
        "moves": [("psychic", 0.915), ("explosion", 0.774), ("gigadrain", 0.625),
                  ("hiddenpowerfire", 0.377), ("sleeppowder", 0.30), ("stunspore", 0.05),
                  ("thief", 0.05)],
        "items": [("leftovers", 0.666), ("miracleberry", 0.239)],
        "teammates": [("snorlax", 0.710), ("cloyster", 0.420), ("zapdos", 0.419)],
    },
    "tyranitar": {
        "usage": 29.36,
        "moves": [("rockslide", 0.810), ("roar", 0.527), ("earthquake", 0.516),
                  ("pursuit", 0.418), ("crunch", 0.20), ("dynamicpunch", 0.15),
                  ("fireblast", 0.12)],
        "items": [("leftovers", 0.929)],
        "teammates": [("snorlax", 0.835), ("zapdos", 0.525), ("cloyster", 0.473)],
    },
    "raikou": {
        "usage": 27.36,
        "moves": [("rest", 0.892), ("sleeptalk", 0.596), ("thunder", 0.586),
                  ("hiddenpowerice", 0.471), ("hiddenpowerwater", 0.25),
                  ("crunch", 0.15)],
        "items": [("leftovers", 0.989)],
        "teammates": [("snorlax", 0.893), ("cloyster", 0.431), ("skarmory", 0.357)],
    },
    "starmie": {
        "usage": 19.99,
        "moves": [("surf", 0.830), ("recover", 0.803), ("rapidspin", 0.714),
                  ("psychic", 0.431), ("thunderwave", 0.20), ("thunderbolt", 0.10)],
        "items": [("leftovers", 0.888), ("miracleberry", 0.063)],
        "teammates": [("snorlax", 0.840), ("tyranitar", 0.390), ("raikou", 0.384)],
    },
    "machamp": {
        "usage": 19.28,
        "moves": [("crosschop", 0.971), ("rockslide", 0.790), ("earthquake", 0.564),
                  ("curse", 0.354), ("hiddenpowerghost", 0.10), ("fireblast", 0.08)],
        "items": [("leftovers", 0.764), ("scopelens", 0.086)],
        "teammates": [("snorlax", 0.820), ("cloyster", 0.314), ("zapdos", 0.313)],
    },
    "skarmory": {
        "usage": 16.76,
        "moves": [("rest", 0.867), ("drillpeck", 0.851), ("whirlwind", 0.713),
                  ("curse", 0.669), ("toxic", 0.10), ("thief", 0.05)],
        "items": [("leftovers", 0.943)],
        "teammates": [("snorlax", 0.839), ("raikou", 0.583), ("starmie", 0.376)],
    },
    "jolteon": {
        "usage": 16.70,
        "moves": [("thunderbolt", 0.85), ("hiddenpowerice", 0.60), ("batonpass", 0.40),
                  ("growth", 0.35), ("thunder", 0.20), ("agility", 0.15)],
        "items": [("leftovers", 0.95)],
        "teammates": [("snorlax", 0.80), ("cloyster", 0.45), ("exeggutor", 0.30)],
    },
    "forretress": {
        "usage": 16.61,
        "moves": [("spikes", 0.95), ("rapidspin", 0.70), ("explosion", 0.60),
                  ("hiddenpowerbug", 0.40), ("toxic", 0.30)],
        "items": [("leftovers", 0.95)],
        "teammates": [("snorlax", 0.80), ("raikou", 0.40), ("machamp", 0.30)],
    },
    "steelix": {
        "usage": 14.32,
        "moves": [("earthquake", 0.90), ("roar", 0.60), ("curse", 0.50),
                  ("explosion", 0.45), ("rest", 0.35), ("irontail", 0.15)],
        "items": [("leftovers", 0.95)],
        "teammates": [("snorlax", 0.80), ("zapdos", 0.40), ("raikou", 0.35)],
    },
    "blissey": {
        "usage": 13.24,
        "moves": [("softboiled", 0.85), ("toxic", 0.60), ("healbell", 0.50),
                  ("lightscreen", 0.35), ("icebeam", 0.30), ("flamethrower", 0.20)],
        "items": [("leftovers", 0.95)],
        "teammates": [("snorlax", 0.75), ("skarmory", 0.45), ("raikou", 0.35)],
    },
    "nidoking": {
        "usage": 12.66,
        "moves": [("earthquake", 0.95), ("icebeam", 0.75), ("thunder", 0.55),
                  ("lovelykiss", 0.45), ("fireblast", 0.20), ("thief", 0.15)],
        "items": [("leftovers", 0.90), ("miracleberry", 0.08)],
        "teammates": [("snorlax", 0.80), ("zapdos", 0.50), ("cloyster", 0.45)],
    },
    "golem": {
        "usage": 12.45,
        "moves": [("earthquake", 0.95), ("explosion", 0.90), ("rapidspin", 0.70),
                  ("roar", 0.55), ("rockslide", 0.15)],
        "items": [("leftovers", 0.90)],
        "teammates": [("snorlax", 0.85), ("zapdos", 0.55), ("gengar", 0.40)],
    },
    "marowak": {
        "usage": 11.41,
        "moves": [("earthquake", 0.95), ("rockslide", 0.80), ("swordsdance", 0.60),
                  ("hiddenpowerbug", 0.30), ("fireblast", 0.20)],
        "items": [("thickclub", 0.95)],
        "teammates": [("snorlax", 0.80), ("zapdos", 0.50), ("cloyster", 0.40)],
    },
}

# sorted by usage for quick access
USAGE_RANKING = sorted(USAGE_STATS.keys(), key=lambda k: -USAGE_STATS[k]["usage"])


def predict_opponent_team(revealed_species: set[str], n_fill: int = 5) -> list[str]:
    """Predict the most likely unrevealed opponent Pokemon.

    Uses Smogon usage stats weighted by teammate correlations with
    what's already been revealed.

    Args:
        revealed_species: set of normalized species names already seen
        n_fill: number of Pokemon to predict

    Returns:
        list of normalized species names, most likely first
    """
    if not revealed_species:
        # no info: just return top usage mons
        return [s for s in USAGE_RANKING if s not in revealed_species][:n_fill]

    # score each candidate by:
    # 1. base usage probability
    # 2. teammate correlation with revealed mons
    candidates = {}
    for species, stats in USAGE_STATS.items():
        if species in revealed_species:
            continue

        score = stats["usage"] / 100.0

        # boost score based on teammate correlation
        for revealed in revealed_species:
            if revealed in USAGE_STATS:
                teammates = USAGE_STATS[revealed].get("teammates", [])
                for tname, tprob in teammates:
                    if tname == species:
                        score *= (1.0 + tprob)  # multiplicative boost

        candidates[species] = score

    ranked = sorted(candidates.keys(), key=lambda k: -candidates[k])
    return ranked[:n_fill]


def get_likely_moveset(species: str) -> list[str]:
    """Get the most likely 4 moves for a species based on usage stats."""
    if species not in USAGE_STATS:
        return ["doubleedge", "earthquake", "rest", "sleeptalk"]  # generic
    moves = USAGE_STATS[species]["moves"]
    return [m for m, _ in sorted(moves, key=lambda x: -x[1])[:4]]


def get_likely_item(species: str) -> str:
    """Get the most likely item for a species."""
    if species not in USAGE_STATS:
        return "leftovers"
    items = USAGE_STATS[species]["items"]
    return items[0][0] if items else "leftovers"
