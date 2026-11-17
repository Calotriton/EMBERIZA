"""Runs the real bacpipe on a folder of recordings, to give the reference that the browser app is compared with.

Run it with bacpipe's own Python environment (it needs torch, onnxruntime, librosa...):

    <bacpipe>/.venv/Scripts/python tools/bacpipe_reference.py <audio folder> [--tasks clustering,probing]
                                                              [--dim umap] [--out tests/reference]

Nothing is written into the bacpipe folder: the model checkpoints go to model/ and the results to
tests/reference/bacpipe_results/, both inside this project. bacpipe's configuration is left at its defaults,
apart from what the options above change.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("audio_dir")
parser.add_argument("--models", default="perch_v2,birdnet_v3")
parser.add_argument("--tasks", default="")
parser.add_argument("--dim", default="umap")
parser.add_argument("--out", default=str(ROOT / "tests" / "reference"))
parser.add_argument("--set", action="append", default=[], help="a bacpipe setting as key=json, repeatable")
args = parser.parse_args()

import bacpipe  # noqa: E402  (slow import, after the arguments are checked)

out = Path(args.out)
overrides = dict(
    audio_dir=str(Path(args.audio_dir).resolve()),
    models=args.models.split(","),
    dim_reduction_model=args.dim,
    evaluation_task=[t for t in args.tasks.split(",") if t],
    overwrite=True,
    main_results_dir=str(out / "bacpipe_results"),
    model_base_path=str(ROOT / "model"),
    device="cpu",
)
for item in args.set:
    key, value = item.split("=", 1)
    overrides[key] = json.loads(value)

# The steps of bacpipe.play(), without its last one (the dashboard server). `dashboard` stays True, as in
# bacpipe's config.yaml: with dashboard=False, bacpipe 1.3.5 stops in plot_embeddings ("points" is not defined).
from bacpipe.core.experiment_manager import replace_default_kwargs_with_user_kwargs, save_logs  # noqa: E402

t0 = time.time()
kwargs = replace_default_kwargs_with_user_kwargs(**overrides)
kwargs["model_base_path"] = bacpipe.ensure_models_exist(Path(kwargs["model_base_path"]), model_names=kwargs["models"])
save_logs(**kwargs)
kwargs["models"] = bacpipe.get_model_names(**kwargs)
loader_dict = bacpipe.run_pipeline_for_models(**kwargs)
bacpipe.model_specific_evaluation(loader_dict, **kwargs)
bacpipe.cross_model_evaluation(**kwargs)
print(f"bacpipe reference finished in {time.time() - t0:.0f} s", file=sys.stderr)
