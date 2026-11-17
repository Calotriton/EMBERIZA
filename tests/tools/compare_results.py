"""Compares two bacpipe result folders for one dataset, file by file: one written by bacpipe itself
(tools/bacpipe_reference.py), the other by EMBERIZA (tests/tools/run_engine.py, or the page).

    python tests/tools/compare_results.py <reference>/bacpipe_results/<dataset> <emberiza>/bacpipe_results/<dataset>

Folder names carry the time of the run and absolute paths differ between machines, so both are left out of the
comparison. Everything else is compared: numbers by their largest difference, text and tables exactly.
`compare(a, b)` returns the findings as a dict, for the tests.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}___")
PATH_KEYS = {"audio_dir", "embed_dir", "main_results_dir", "model_base_path"}   # absolute, so they differ by machine


def tree(root):
    """{path without the run's time stamp: file}"""
    out = {}
    for f in sorted(Path(root).rglob("*")):
        if f.is_file() and "logs" not in f.relative_to(root).parts[:1]:
            parts = [STAMP.sub("", p) for p in f.relative_to(root).parts]
            out["/".join(parts)] = f
    return out


def _numbers(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    if a.size == 0:
        return {"max_abs_diff": 0.0}
    return {"max_abs_diff": float(np.nanmax(np.abs(a - b))), "identical": bool(np.array_equal(a, b))}


def _strip_paths(d):
    """Leave out absolute paths, and write the relative ones with "/" (bacpipe on Windows writes "\\")."""
    if isinstance(d, dict):
        return {k: _strip_paths(v) for k, v in d.items() if k not in PATH_KEYS}
    if isinstance(d, list):
        return [_strip_paths(v) for v in d]
    if isinstance(d, str):
        return d.replace("\\", "/")
    return d


def _table(a, b, sep):
    da, db = pd.read_csv(a, sep=sep), pd.read_csv(b, sep=sep)
    if list(da.columns) != list(db.columns):
        return {"columns_differ": [sorted(set(da.columns) ^ set(db.columns))[:8]]}
    if len(da) != len(db):
        return {"rows": [len(da), len(db)]}
    out, worst = {}, 0.0
    for col in da.columns:
        if pd.api.types.is_numeric_dtype(da[col]) and pd.api.types.is_numeric_dtype(db[col]):
            worst = max(worst, float(np.nanmax(np.abs(da[col].to_numpy(float) - db[col].to_numpy(float)))) if len(da) else 0.0)
        elif not (da[col].astype(str).str.replace("\\", "/") == db[col].astype(str).str.replace("\\", "/")).all():
            out.setdefault("text_columns_differ", []).append(col)
    out["max_abs_diff"] = worst
    out["rows_compared"] = len(da)
    return out


def _json(a, b):
    ja, jb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    flat_a, flat_b = {}, {}

    def flatten(obj, prefix, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in PATH_KEYS:
                    flatten(v, f"{prefix}/{k}", out)
        elif isinstance(obj, list) and obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            out[prefix] = np.asarray(obj, dtype=float)
        else:
            out[prefix] = obj
    flatten(ja, "", flat_a)
    flatten(jb, "", flat_b)
    out = {}
    if set(flat_a) != set(flat_b):
        out["keys_differ"] = sorted(set(flat_a) ^ set(flat_b))[:8]
    worst, other = 0.0, []
    for key in set(flat_a) & set(flat_b):
        va, vb = flat_a[key], flat_b[key]
        if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
            if va.shape != vb.shape:
                other.append(key)
            elif va.size:
                worst = max(worst, float(np.abs(va - vb).max()))
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            worst = max(worst, abs(va - vb))
        elif _strip_paths(va) != _strip_paths(vb):
            other.append(key)
    out["max_abs_diff"] = worst
    if other:
        out["values_differ"] = sorted(other)[:8]
    return out


def compare_file(key, a, b):
    suffix = a.suffix.lower()
    if suffix == ".npy":
        xa, xb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        if xa.dtype == object:                                   # a dictionary of label arrays (clusterings)
            da, db = xa.item(), xb.item()
            same = set(da) == set(db) and all(len(da[k]) == len(db[k]) for k in da)
            return {"keys": sorted(da), "same_keys_and_lengths": same}
        return _numbers(xa, xb)
    if suffix in (".yml", ".yaml"):
        ya, yb = _strip_paths(yaml.safe_load(a.read_text())), _strip_paths(yaml.safe_load(b.read_text()))
        fa = {k: str(v).replace("\\", "/") for k, v in pd.json_normalize(ya, sep="/").iloc[0].items()}
        fb = {k: str(v).replace("\\", "/") for k, v in pd.json_normalize(yb, sep="/").iloc[0].items()}
        return {"identical": fa == fb, **({} if fa == fb else {"differ": sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))[:8]})}
    if suffix == ".json":
        return _json(a, b)
    if suffix == ".csv":
        return _table(a, b, ",")
    if suffix == ".txt":
        return _table(a, b, "\t")
    return {"skipped": "not compared (a picture or a binary file)"}


def compare(reference, mine):
    ta, tb = tree(reference), tree(mine)
    report = {"only_in_reference": sorted(set(ta) - set(tb)), "only_in_emberiza": sorted(set(tb) - set(ta)), "files": {}}
    for key in sorted(set(ta) & set(tb)):
        try:
            report["files"][key] = compare_file(key, ta[key], tb[key])
        except Exception as e:
            report["files"][key] = {"error": f"{type(e).__name__}: {e}"}
    return report


def summary(report):
    lines = []
    groups = {}
    for key, result in report["files"].items():
        parts = key.split("/")
        group = parts[0] + ("/" + parts[2] if parts[0] == "evaluations" and len(parts) > 3 else "") + " *" + Path(key).suffix
        groups.setdefault(group, []).append((key, result))
    for group, items in sorted(groups.items()):
        worst = max((r.get("max_abs_diff", 0.0) for _, r in items), default=0.0)
        problems = [(k, r) for k, r in items if any(x in r for x in ("error", "shape", "rows", "columns_differ", "keys_differ",
                                                                      "values_differ", "text_columns_differ", "differ"))
                    or r.get("identical") is False and "max_abs_diff" not in r or r.get("same_keys_and_lengths") is False]
        identical = sum(1 for _, r in items if r.get("identical") or r.get("max_abs_diff") == 0.0)
        lines.append(f"{group:55s} {len(items):3d} files  identical {identical:3d}  max |diff| {worst:.3g}"
                     + (f"  PROBLEMS {len(problems)}" if problems else ""))
        for k, r in problems[:6]:
            lines.append(f"      {k}: {json.dumps(r)[:300]}")
    lines.append(f"only in the reference ({len(report['only_in_reference'])}): {report['only_in_reference'][:12]}")
    lines.append(f"only in EMBERIZA ({len(report['only_in_emberiza'])}): {report['only_in_emberiza'][:12]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(compare(sys.argv[1], sys.argv[2])))
