import pytest

from sfxrender import SfxUriError, parse_sfx_uri


def test_parse_sfx_uri() -> None:
    spec = parse_sfx_uri("sfx:impact.knock?material=oak&count=3&seed=42")
    assert spec.effect == "impact.knock"
    assert spec.parameters["material"] == "oak"
    assert spec.parameters["count"] == "3"
    assert spec.seed == 42


def test_rejects_wrong_scheme() -> None:
    with pytest.raises(SfxUriError):
        parse_sfx_uri("https://example.com/knock.wav")


def test_rejects_duplicate_parameter() -> None:
    with pytest.raises(SfxUriError):
        parse_sfx_uri("sfx:impact.knock?count=2&count=3")
