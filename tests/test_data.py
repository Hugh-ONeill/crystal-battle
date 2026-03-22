import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"


def test_pokemon_json_exists():
    path = DATA_DIR / "pokemon.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 251


def test_pokemon_have_required_fields():
    data = json.loads((DATA_DIR / "pokemon.json").read_text())
    for p in data:
        assert "id" in p
        assert "name" in p
        assert "base_stats" in p
        assert "types" in p
        assert "learnset" in p
        assert "hp" in p["base_stats"]
        assert "attack" in p["base_stats"]
        assert "speed" in p["base_stats"]


def test_moves_json_exists():
    path = DATA_DIR / "moves.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) > 200


def test_moves_damage_class():
    """Gen 2 damage class should be based on type, not per-move."""
    data = json.loads((DATA_DIR / "moves.json").read_text())
    physical_types = {"normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel"}
    special_types = {"fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"}

    for m in data:
        if m["power"] > 0:
            if m["type"] in physical_types:
                assert m["damage_class"] == "physical", f"{m['name']} should be physical"
            elif m["type"] in special_types:
                assert m["damage_class"] == "special", f"{m['name']} should be special"


def test_type_chart_json():
    data = json.loads((DATA_DIR / "type_chart.json").read_text())
    assert len(data) == 17
    assert "fairy" not in data


def test_pokemon_have_learnsets():
    data = json.loads((DATA_DIR / "pokemon.json").read_text())
    # most pokemon should have at least 1 move
    with_moves = sum(1 for p in data if len(p["learnset"]) > 0)
    assert with_moves > 240  # allow a few edge cases
