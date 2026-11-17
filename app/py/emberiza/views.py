"""What the page shows, read from bacpipe's result files with bacpipe's own loaders.

Everything here works from the results folder alone, so results made earlier, or by bacpipe itself on another
computer, open the same way. Only the spectrogram needs the recordings.
"""
import json
from pathlib import Path

import numpy as np

from .engine import _SPLIT_OUT, _without

import bacpipe                                                                    # noqa: E402
import librosa as lb                                                              # noqa: E402
from bacpipe.core.experiment_manager import replace_default_kwargs_with_user_kwargs   # noqa: E402
from bacpipe.embedding_evaluation import label_embeddings as le                  # noqa: E402
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import EmbedAndLabelLoader   # noqa: E402
from scipy.signal.windows import tukey                                            # noqa: E402

NOT_LABELS = {"start", "end"}


class Views:
    def __init__(self, engine):
        self.engine = engine
        self._points, self._json = {}, {}

    def _kwargs(self):
        kwargs = _without(self.engine.kwargs, "models", *_SPLIT_OUT)
        return replace_default_kwargs_with_user_kwargs(remove_keys=list(_SPLIT_OUT), **kwargs)

    def _paths(self, model):
        k = self._kwargs()
        return le.make_set_paths_func(self.engine.kwargs["audio_dir"], **k)(model)

    def _read_json(self, path):
        path = Path(path)
        key = (str(path), path.stat().st_mtime)
        if key not in self._json:
            self._json[key] = json.loads(path.read_text())
        return self._json[key]

    # ------------------------------------------------------------------ overview

    def summary(self):
        """One entry per model: what was embedded, and which evaluations exist."""
        kw = self.engine.kwargs
        out = {"dataset": Path(kw["audio_dir"]).stem, "dim_reduction_model": kw["dim_reduction_model"],
               "evaluation_task": list(kw.get("evaluation_task") or []), "failed": self.engine.failed, "models": []}
        for model, loader in self.engine.loaders.items():
            md, paths = loader.metadata_dict, self._paths(model)
            probes = {p.stem.replace("probe_results_", ""): self._read_json(p)
                      for p in sorted(paths.probe_path.glob("probe_results_*.json"))}
            clust = paths.clust_path / "clust_results.json"
            out["models"].append({
                "name": model, "files": len(md["files"]["audio_files"]),
                "embeddings": int(sum(md["files"]["nr_embeds_per_file"])),
                "embedding_size": md.get("embedding_size"), "sample_rate": md.get("sample_rate (Hz)"),
                "segment_s": md["segment_length (samples)"] / md["sample_rate (Hz)"],
                "seconds": float(sum(md["files"]["file_lengths (s)"])),
                "has_predictions": (paths.preds_path / "original_classifier_outputs").exists(),
                "clustering": self._read_json(clust) if clust.exists() else None,
                "probing": {name: {k: v for k, v in result.items() if k != "config"} for name, result in probes.items()},
            })
        return json.dumps(out, default=float)

    # ------------------------------------------------------------------ the embedding plot

    def points(self, model):
        """The reduced embeddings of one model with every label bacpipe offers for them (its `label_by` list).

        Loaded with bacpipe's EmbedAndLabelLoader, which also writes the label files if they are missing.
        """
        if model not in self._points:
            k = self._kwargs()
            self._paths(model)                                    # sets the paths bacpipe's loaders read
            vis = EmbedAndLabelLoader(dim_reduction_model=self.engine.kwargs["dim_reduction_model"], **k)
            labels, embeds, _ = vis.get_data(model, "audio_file_name")
            n = len(embeds["x"])
            data = {"model": model, "n": n, "x": embeds["x"], "y": embeds["y"], "z": embeds.get("z"),
                    "file": list(embeds["metadata"]["audio_files"]), "start": [float(v) for v in labels["start"]],
                    "end": [float(v) for v in labels["end"]],
                    "annotated": (~np.asarray(vis.bool_noise[model])).tolist(),
                    "labels": {key: [str(v) for v in values] for key, values in labels.items()
                               if key not in NOT_LABELS and len(values) == n}}
            self._points[model] = data
        return json.dumps(self._points[model])

    def forget(self):
        self._points, self._json = {}, {}

    # ------------------------------------------------------------------ one window: sound and spectrogram

    def segment(self, model, index):
        """The sound of one embedding and its spectrogram, made as bacpipe's dashboard makes them:
        librosa.load at the model's sample rate, stft with a 1024-sample window, decibels relative to the maximum."""
        pts = json.loads(self.points(model))
        md = self.engine.loaders[model].metadata_dict
        sr, segment_length = md["sample_rate (Hz)"], md["segment_length (samples)"]
        start, end, file = pts["start"][index], pts["end"][index], pts["file"][index]
        path = Path(self.engine.kwargs["audio_dir"]) / file
        if not path.exists():
            return {"missing": file}
        audio, _ = lb.load(path, sr=sr, offset=float(start), duration=float(end) - float(start))
        if (float(end) - float(start)) * sr < segment_length or len(audio) < segment_length:
            audio = (tukey(len(audio), alpha=0.01) * audio).astype(np.float32)
            audio = lb.util.fix_length(audio, size=segment_length, mode=self.engine.kwargs["padding"])
        spec = lb.amplitude_to_db(np.abs(lb.stft(audio, win_length=1024)), ref=np.max).astype(np.float32)
        return {"file": file, "start": float(start), "end": float(end), "sr": int(sr),
                "audio": np.ascontiguousarray(audio, dtype=np.float32), "spec": np.ascontiguousarray(spec).reshape(-1),
                "rows": int(spec.shape[0]), "cols": int(spec.shape[1]), "f_max": sr / 2,
                "labels": {k: v[index] for k, v in pts["labels"].items()},
                "predictions": self._predictions_at(model, file, start)}

    def _predictions_at(self, model, file, start):
        paths, md = self._paths(model), self.engine.loaders[model].metadata_dict
        rel = Path(file)
        f = paths.preds_path / "original_classifier_outputs" / rel.parent / f"{rel.stem}_{model}.json"
        if not f.exists():
            return []
        time_bin = int(round(start / (md["segment_length (samples)"] / md["sample_rate (Hz)"])))
        found = []
        for species, d in self._read_json(f).items():
            if species != "head" and time_bin in d["time_bins_exceeding_threshold"]:
                found.append([species, d["classifier_predictions"][d["time_bins_exceeding_threshold"].index(time_bin)]])
        found.sort(key=lambda item: -item[1])
        return found[: int(self.engine.kwargs.get("nr_predictions_to_display", 3))]

    # ------------------------------------------------------------------ the classifier's predictions

    def predictions(self, model):
        """Every saved prediction of the model's own classifier, with the time of each window, for the heatmap.

        bacpipe saves the predictions above `classifier_threshold`. The page can only raise that threshold.
        """
        loader = self.engine.loaders[model]
        array, keys = loader.get_preds_array(return_type="array")
        if array is None:
            return json.dumps(None)
        labels = le.metadata_labels(model=model, paths=self._paths(model), overwrite=False, return_type="dict",
                                    **_without(self._kwargs(), "overwrite", "paths"))
        rows, cols = np.nonzero(array)
        return json.dumps({"species": list(keys), "bin": rows.tolist(), "class": cols.tolist(),
                           "score": [round(float(v), 5) for v in array[rows, cols]],
                           "time": list(labels["continuous_timestamp"]), "file": list(labels["audio_file_name"]),
                           "threshold": float(self.engine.kwargs["classifier_threshold"]), "bins": int(array.shape[0])})
