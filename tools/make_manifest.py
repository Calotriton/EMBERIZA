"""Writes app/py/manifest.json: the list of Python and data files the page copies into Pyodide when it starts.

    python tools/make_manifest.py

Run it after adding, removing or renaming a file under app/py/. tests/test_vendored.py fails when it is out of date.
"""
import json
from pathlib import Path

PY = Path(__file__).resolve().parent.parent / "app" / "py"
SKIP_NAMES = {"manifest.json"}
SKIP_PARTS = {"__pycache__"}


def files():
    return sorted(p.relative_to(PY).as_posix() for p in PY.rglob("*")
                  if p.is_file() and p.name not in SKIP_NAMES and not SKIP_PARTS & set(p.parts) and p.suffix != ".pyc")


if __name__ == "__main__":
    found = files()
    (PY / "manifest.json").write_text(json.dumps({"files": found}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(found)} files, {sum((PY / f).stat().st_size for f in found) / 1e6:.1f} MB")
