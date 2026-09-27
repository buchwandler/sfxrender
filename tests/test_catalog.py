from sfxrender import catalog


def test_catalog_contains_builtin_effects() -> None:
    values = catalog()
    assert "impact.knock" in values
    assert "footsteps.walk" in values
    assert "door.open" in values
    assert "door.close" in values
    assert "printer.print" in values
    assert "printer.tray_open" in values
    assert "printer.power_switch" in values
    assert "printer.restart" in values
    assert "pen.write" in values
    assert "printer.wake" in values
    assert "parameters" in values["impact.knock"]


def test_catalog_returns_copy() -> None:
    first = catalog()
    first.clear()
    assert "impact.knock" in catalog()


def test_renderer_and_catalog_stay_in_sync_and_defaults_render() -> None:
    from urllib.parse import urlencode

    from sfxrender import SFXRenderer

    renderer = SFXRenderer()
    effects = catalog()
    assert set(renderer.effects()) == set(effects)

    for effect, descriptor in effects.items():
        parameters = {
            name: str(parameter["default"])
            for name, parameter in descriptor["parameters"].items()
            if "default" in parameter
        }
        query = urlencode(parameters)
        uri = f"sfx:{effect}" + (f"?{query}" if query else "")
        rendered = renderer.render_uri(uri)
        assert rendered.samples.size > 0
