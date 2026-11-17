"""EMBERIZA's engine against bacpipe itself, on real recordings.

Needs what is not in the repository: the model files in model/ and a reference made by tools/bacpipe_reference.py
in tests/reference/. The folder of recordings is given as EMBERIZA_AUDIO (default: none, and the tests are skipped).
The real onnxruntime plays the part of ONNX Runtime Web, so this checks everything but the browser's own arithmetic,
which tests/tools/browser_e2e.py covers.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIO = os.environ.get("EMBERIZA_AUDIO")
REFERENCE = ROOT / "tests" / "reference" / "bacpipe_results" / (Path(AUDIO).name if AUDIO else "-")
ready = AUDIO and REFERENCE.exists() and (ROOT / "model" / "perch_v2" / "perch_v2_no_dft.onnx").exists()
pytestmark = pytest.mark.skipif(not ready, reason="needs EMBERIZA_AUDIO, model/ and tests/reference/ (see the module's docstring)")

sys.path.insert(0, str(ROOT / "tests" / "tools"))


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    from compare_results import compare
    from run_engine import run
    out = tmp_path_factory.mktemp("emberiza") / "bacpipe_results"
    has_annotations = (Path(AUDIO) / "annotations.csv").exists()
    engine = run(AUDIO, out, dim="pca", tasks=["clustering", "probing"] if has_annotations else [], log=lambda *_: None)
    assert engine.failed == []
    return compare(REFERENCE, out / Path(AUDIO).name)


def _files(report, part, suffix):
    return {k: v for k, v in report["files"].items() if part in k and k.endswith(suffix)}


def test_embeddings_are_bacpipes_to_the_last_bit(report):
    files = _files(report, "embeddings/", ".npy")
    assert files and all(v.get("identical") for k, v in files.items() if "dim_reduced" not in k), files


def test_metadata_and_labels_are_identical(report):
    for key, result in {**_files(report, "embeddings/", ".yml"), **_files(report, "/labels/", ".csv")}.items():
        if "dim_reduced" not in key:
            assert result.get("identical") or result.get("max_abs_diff") == 0.0, (key, result)


def test_predictions_agree_to_the_fifth_decimal(report):
    files = {**_files(report, "/predictions/", ".json"), **_files(report, "/predictions/", ".csv")}
    assert files and all(v.get("max_abs_diff", 1) < 1e-5 and "values_differ" not in v for v in files.values()), files


def test_no_result_file_is_missing_except_the_pictures(report):
    missing = [f for f in report["only_in_reference"] if not f.endswith(".png") and "dim_reduced" not in f]
    assert missing == []
    assert [f for f in report["only_in_emberiza"] if "dim_reduced" not in f] == []
    assert np.all([f.endswith(".png") or "umap" in f for f in report["only_in_reference"]])
