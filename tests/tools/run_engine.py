"""Runs EMBERIZA's engine outside the browser, on the package versions Pyodide provides. The real onnxruntime plays
the part of ONNX Runtime Web: it is the only thing here that the page does differently.

    python tests/tools/run_engine.py <audio folder> <results folder> [--dim umap] [--tasks clustering,probing]
                                     [--batch 4] [--set key=json ...]

Used by the tests, and by hand to compare with tools/bacpipe_reference.py (see tests/tools/compare_results.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "py"))


def _real_onnxruntime():
    """The installed onnxruntime, even when the engine has already claimed the name for its replay (as it has when
    other tests ran first in the same process). The replay keeps the name afterwards."""
    claimed = sys.modules.pop("onnxruntime", None)
    try:
        import onnxruntime as real
    finally:
        if claimed is not None:
            sys.modules["onnxruntime"] = claimed
    return real


real_onnxruntime = _real_onnxruntime()


def run(audio_dir, results_dir, models=("perch_v2", "birdnet_v3"), dim="umap", tasks=(), batch=None, settings=None,
        model_dir=ROOT / "model", log=print):
    from emberiza.engine import MODELS, Engine

    engine = Engine(model_base_path=Path(results_dir).parent / "model_markers")
    config = {"audio_dir": str(Path(audio_dir).resolve()), "models": list(models), "dim_reduction_model": dim,
              "evaluation_task": list(tasks), "overwrite": True}
    engine.configure(config, {"main_results_dir": str(results_dir), "device": "cpu", **(settings or {})})
    for model in engine.kwargs["models"]:
        info = engine.start_model(model)
        if not info["already_computed"]:
            session = real_onnxruntime.InferenceSession(str(Path(model_dir) / MODELS[model]["file"]),
                                                        providers=["CPUExecutionProvider"])
            size = batch or info["batch_size"]
            t0, windows = time.time(), 0
            for i, name in enumerate(info["files"]):
                frames = engine.prepare(i)
                if frames is None:
                    continue
                parts = [session.run(info["wanted"], {info["input"]: frames[s:s + size]}) for s in range(0, len(frames), size)]
                outputs = {name_: np.concatenate([p[k] for p in parts]) for k, name_ in enumerate(info["wanted"])}
                windows += engine.store(i, outputs)
            log(f"{model}: {windows} windows in {time.time() - t0:.1f} s, batches of {size}")
            engine.finish_model()
        engine.reduce(model)
    engine.evaluate()
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir")
    parser.add_argument("results_dir")
    parser.add_argument("--models", default="perch_v2,birdnet_v3")
    parser.add_argument("--dim", default="umap")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    extra = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    engine = run(args.audio_dir, args.results_dir, args.models.split(","), args.dim,
                 [t for t in args.tasks.split(",") if t], args.batch, extra)
    if engine.failed:
        print("FAILED RECORDINGS:", json.dumps(engine.failed, indent=1))
