"""The copy of bacpipe is unchanged, the page's file list is current, and every bacpipe option is in the form."""
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "app" / "py"
sys.path.insert(0, str(ROOT / "tools"))


def test_bacpipe_files_are_unchanged():
    record = json.loads((PY / "bacpipe" / "VENDORED.json").read_text())
    assert record["modified"] is False
    for name, sha in record["files"].items():
        assert hashlib.sha256((PY / "bacpipe" / name).read_bytes()).hexdigest() == sha, f"{name} was edited"
    on_disk = {p.relative_to(PY / "bacpipe").as_posix() for p in (PY / "bacpipe").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts} - {"VENDORED.json"}
    assert on_disk == set(record["files"]), "files were added to or removed from the copy of bacpipe"


def test_manifest_is_current():
    import make_manifest
    assert json.loads((PY / "manifest.json").read_text())["files"] == make_manifest.files(), "run tools/make_manifest.py"


def _flat(d, prefix=""):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from _flat(value, f"{prefix}{key}.")
        else:
            yield prefix + key


def test_every_bacpipe_option_is_in_the_page():
    source = (ROOT / "app" / "options.js").read_text(encoding="utf-8")
    fields = set(re.findall(r"key: '([^']+)'", source))
    elsewhere = set(re.findall(r"(\w+): '", source.split("CONFIG_ELSEWHERE = {")[1].split("}")[0]))
    settings = yaml.safe_load((PY / "bacpipe" / "settings.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((PY / "bacpipe" / "config.yaml").read_text(encoding="utf-8"))
    # the name of a probe or clustering method and the probe's table name identify a block: they are not choices
    fixed = {k for k in _flat(settings) if k.endswith(".name") or k.endswith(".dataset_csv_path")}
    assert set(_flat(settings)) - fixed - fields == set()
    assert set(config) - fields - elsewhere == set()
    assert fields - set(_flat(settings)) - set(config) == {"__backend"}, "the form has a field bacpipe does not know"
