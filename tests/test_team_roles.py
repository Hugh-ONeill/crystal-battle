"""Team-level derivation: properties no per-species entry can carry.

Guards the three things the pass exists for — sole-role escalation, field
resource chains (and the honesty rule that an UNENTERED teammate makes an
apparent orphan unverifiable rather than a build error), and entry economics.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.team_roles import analyze, load_roles, resource_tokens, satisfies


@pytest.fixture(scope="module")
def roles():
    return load_roles()


def write(tmp_path, name, mons):
    p = tmp_path / f"{name}.txt"
    p.write_text("\n\n".join(mons))
    return str(p)


def test_resource_tokens_normalises_free_prose():
    assert resource_tokens("SUN — Chlorophyll doubles its Speed") == {"sun"}
    assert resource_tokens("Trick Room turns") == {"trickroom"}
    assert resource_tokens("a TERRAIN its seed can consume") == {"terrain"}
    # a specific terrain must not also register as the generic one
    assert resource_tokens("grassy terrain") == {"grassyterrain"}
    # not a field resource at all
    assert resource_tokens("a free turn to activate Flame Orb") == set()


def test_generic_terrain_is_satisfied_by_any_specific_one():
    assert satisfies({"electricterrain"}, "terrain")
    assert satisfies({"sun"}, "sun")
    assert not satisfies({"sun"}, "rain")


def test_sole_role_is_detected_and_a_second_holder_clears_it(tmp_path, roles):
    one = write(tmp_path, "one", ["Great Tusk @ Leftovers", "Kingambit @ Leftovers"])
    assert "greattusk" in analyze(one, roles)["sole"]
    two = write(tmp_path, "two", ["Great Tusk @ Leftovers", "Corviknight @ Leftovers"])
    a = analyze(two, roles)
    assert len(a["coverage"]["hazard-remover"]) == 2
    assert "hazard-remover" not in a["sole"].get("greattusk", [])


def test_resource_chain_links_setter_to_dependent(tmp_path, roles):
    t = write(tmp_path, "sun", ["Torkoal @ Heat Rock", "Venusaur @ Life Orb"])
    a = analyze(t, roles)
    assert "sun" in a["provides"]["torkoal"]
    assert "venusaur" in a["dependents"]["torkoal"]
    assert not a["orphans"]


def test_missing_provider_is_a_build_error_only_when_all_are_entered(tmp_path, roles):
    orphan = write(tmp_path, "orphan", ["Venusaur @ Life Orb", "Kingambit @ Leftovers"])
    a = analyze(orphan, roles)
    assert a["orphans"] and not a["unknown"]      # confirmed build error
    # an unentered teammate could be the provider — Pincurchin was exactly
    # this case, so the verdict must degrade to "unverifiable"
    unknown = write(tmp_path, "unk", ["Venusaur @ Life Orb", "Fakemon @ Leftovers"])
    b = analyze(unknown, roles)
    assert b["orphans"] and b["unknown"]


def test_entry_enablers_are_found_from_the_paste(tmp_path, roles):
    t = write(tmp_path, "pivot", [
        "Great Tusk @ Leftovers\n- Rapid Spin",
        "Slowking-Galar @ Heavy-Duty Boots\n- Chilly Reception"])
    a = analyze(t, roles)
    assert "slowkinggalar" in a["slow_pivots"]
    assert any(g["species"] == "greattusk" for g in a["entry_gated"])


def test_healing_wish_is_recorded_as_a_restorer(tmp_path, roles):
    t = write(tmp_path, "hw", ["Enamorus @ Choice Scarf\n- Healing Wish",
                               "Ceruledge @ Life Orb"])
    a = analyze(t, roles)
    assert any(r["move"] == "healingwish" for r in a["restorers"])


def test_wincon_outlook_names_the_blockers_and_the_window(tmp_path, roles):
    """A wincon is not live until its answers are gone — the blockers must be
    named by class, since each defeats a setup plan differently."""
    from showdown.team_roles import wincon_outlook
    t = write(tmp_path, "w", ["Zamazenta @ Leftovers", "Great Tusk @ Leftovers"])
    a = analyze(t, roles)
    rows = wincon_outlook(a, ["dondozo", "heatran", "dragapult"], roles)
    zama = next(r for r in rows if r["species"] == "zamazenta")
    assert zama["is_wincon"] and zama["sole_wincon"]
    classes = {b["class"] for b in zama["blockers"]}
    assert "anti-setup" in classes      # Dondozo's Unaware
    assert "trapper" in classes         # Heatran's Magma Storm
    # a species with no blocking role contributes nothing
    assert "dragapult" not in {b["species"] for b in zama["blockers"]}


def test_open_window_when_nothing_blanks_it(tmp_path, roles):
    from showdown.team_roles import wincon_outlook, wincon_report
    t = write(tmp_path, "w2", ["Zamazenta @ Leftovers"])
    rows = wincon_outlook(analyze(t, roles), ["dragapult", "greattusk"], roles)
    assert not rows[0]["blockers"]
    assert any("window is OPEN" in ln for ln in wincon_report(rows))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
