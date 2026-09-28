from sfxrender import SFXRenderer, catalog, llm_catalog


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


def test_every_builtin_has_llm_authoring_metadata_and_parameter_descriptions() -> None:
    for descriptor in catalog().values():
        assert descriptor["description"].strip()
        assert descriptor["parameters"]
        assert descriptor["llm"]["use_when"]
        assert "avoid_when" in descriptor["llm"]
        assert descriptor["llm"]["examples"]
        for parameter in descriptor["parameters"].values():
            assert parameter["description"].strip()


def test_all_llm_examples_parse_and_validate_against_their_catalog_effect() -> None:
    from sfxrender import SFXRenderer

    renderer = SFXRenderer()
    for effect, descriptor in renderer.catalog().items():
        for uri in descriptor["llm"]["examples"]:
            assert renderer.validate_uri(uri).effect == effect


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


def test_runtime_llm_manifest_has_expected_schema_rules_and_effects() -> None:
    renderer = SFXRenderer()
    manifest = renderer.llm_catalog()

    assert manifest["schema"] == "sfxrender.llm-catalog.v1"
    assert manifest["uri_template"]
    assert manifest["ssmd_template"]
    assert manifest["rules"]
    assert (
        "If no listed effect matches the audible event, do not invent an effect."
        in manifest["rules"]
    )
    assert (
        "Use a short sound description in an SSMD audio span, not replacement dialogue."
        in manifest["rules"]
    )
    assert set(manifest["effects"]) == set(catalog())
    assert list(manifest["effects"]) == sorted(manifest["effects"])
    assert "printer.print" in manifest["effects"]
    assert "pen.write" in manifest["effects"]


def test_llm_manifests_are_defensive_and_module_runtime_equivalent() -> None:
    first = llm_catalog()
    first["effects"].clear()
    first["rules"].clear()
    assert llm_catalog()["effects"]
    assert llm_catalog()["rules"]

    assert llm_catalog()["effects"] == SFXRenderer().llm_catalog()["effects"]
