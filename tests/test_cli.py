from __future__ import annotations

import json

from sfxrender import catalog
from sfxrender.cli import main


def test_cli_llm_catalog_outputs_manifest_json(capsys) -> None:
    assert main(["--llm-catalog"]) == 0
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["schema"] == "sfxrender.llm-catalog.v1"
    assert "printer.print" in manifest["effects"]
    assert "pen.write" in manifest["effects"]


def test_cli_catalog_keeps_builtin_catalog_output(capsys) -> None:
    assert main(["--catalog"]) == 0
    assert json.loads(capsys.readouterr().out) == catalog()
