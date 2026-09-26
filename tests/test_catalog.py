from sfxrender import catalog


def test_catalog_contains_builtin_effects() -> None:
    values = catalog()
    assert "impact.knock" in values
    assert "footsteps.walk" in values
    assert "parameters" in values["impact.knock"]


def test_catalog_returns_copy() -> None:
    first = catalog()
    first.clear()
    assert "impact.knock" in catalog()
