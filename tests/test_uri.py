import pytest

from sfxrender import SFXURIError, SfxUriError, parse_sfx_uri


def test_parse_sfx_uri() -> None:
    spec = parse_sfx_uri("sfx:impact.knock?material=oak&count=3&seed=42")
    assert spec.effect == "impact.knock"
    assert spec.parameters["material"] == "oak"
    assert spec.parameters["count"] == "3"
    assert spec.seed == 42


def test_uri_error_alias_remains_compatible() -> None:
    assert SfxUriError is SFXURIError


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/knock.wav",
        "sfx:",
        "sfx:impact.knock?count",
        "sfx:impact.knock?count=%GG",
        "sfx:impact.knock#fragment",
    ],
)
def test_invalid_uri_inputs_raise_public_error(uri: str) -> None:
    with pytest.raises(SFXURIError):
        parse_sfx_uri(uri)


def test_rejects_duplicate_parameter() -> None:
    with pytest.raises(SFXURIError, match="duplicate query parameter: count"):
        parse_sfx_uri("sfx:impact.knock?count=2&count=3")
