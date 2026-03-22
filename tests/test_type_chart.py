import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.types import TypeChart


def test_load():
    tc = TypeChart.load()
    assert len(tc.types) == 17
    assert "fairy" not in tc.types
    assert "normal" in tc.types


def test_basic_effectiveness():
    tc = TypeChart.load()
    # fire -> grass = 2x
    assert tc.effectiveness("fire", "grass") == 2.0
    # water -> fire = 2x
    assert tc.effectiveness("water", "fire") == 2.0
    # normal -> ghost = 0x
    assert tc.effectiveness("normal", "ghost") == 0.0
    # electric -> ground = 0x
    assert tc.effectiveness("electric", "ground") == 0.0


def test_gen2_corrections():
    tc = TypeChart.load()
    # ghost -> steel: resisted in gen 2
    assert tc.effectiveness("ghost", "steel") == 0.5
    # dark -> steel: resisted in gen 2
    assert tc.effectiveness("dark", "steel") == 0.5


def test_combined_effectiveness():
    tc = TypeChart.load()
    # ice -> dragon/flying = 4x
    assert tc.combined_effectiveness("ice", ["dragon", "flying"]) == 4.0
    # ground -> fire/flying: 2x * 0x = 0x
    assert tc.combined_effectiveness("ground", ["fire", "flying"]) == 0.0
    # normal -> rock/steel: 0.5 * 0.5 = 0.25
    assert tc.combined_effectiveness("normal", ["rock", "steel"]) == 0.25


def test_neutral():
    tc = TypeChart.load()
    # normal -> normal = 1x
    assert tc.effectiveness("normal", "normal") == 1.0
