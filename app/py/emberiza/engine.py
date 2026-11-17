"""Runs bacpipe step by step, so that the page can do the one thing Python cannot do in a browser: wait for the
neural network.

bacpipe.play() does this for each model:

    run_pipeline_for_models -> run_pipeline_for_single_model -> generate_embeddings
        Loader, Embedder, then for each recording: prepare_audio, the model, save        <- the model is the gap
        the classifier's annotation table, metadata.yml
        generate_embeddings again for the dimensionality reduction
    model_specific_evaluation, cross_model_evaluation

The Engine below makes the same calls on bacpipe's own objects, in the same order, with the same arguments. The loop
over recordings is cut in two around the model: `prepare` returns the audio windows bacpipe made, the page runs the
network on them, and `store` lets bacpipe carry on with the results (see replay.py). Everything else is one
call into bacpipe. The comments name the bacpipe function each block comes from.
"""
import logging
import time
from pathlib import Path

import numpy as np

from . import replay, runtime

runtime.install()

import bacpipe                                                    # noqa: E402
import torch                                                      # noqa: E402
from bacpipe.core import workflows                                # noqa: E402
from bacpipe.core.experiment_manager import Loader, replace_default_kwargs_with_user_kwargs, save_logs   # noqa: E402
from bacpipe.embedding_evaluation.label_embeddings import make_set_paths_func   # noqa: E402
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import EmbedAndLabelLoader, plot_embeddings   # noqa: E402
from bacpipe.model_pipelines.runner import Embedder               # noqa: E402

logger = logging.getLogger("bacpipe")

# The two models, and what their files take and return (tests/test_models.py reads this from the files themselves).
# `wanted` are the outputs bacpipe uses. Perch 2.0's file also returns a spectrogram and a spatial embedding,
# which bacpipe reads and drops, so the page does not ask for them.
MODELS = {
    "perch_v2": {
        "title": "Perch 2.0", "file": "perch_v2/perch_v2_no_dft.onnx", "bytes": 413350933,
        "sha256": "4dcf71c18a147198545944bb5149697e89e3ad2e16637fa8f0edf6d13035a017",
        "input": "inputs", "outputs": ["embedding", "spatial_embedding", "spectrogram", "label"],
        "wanted": ["embedding", "label"],
    },
    "birdnet_v3": {
        "title": "BirdNET+ V3.0 (preview 3.1)", "file": "birdnet_v3/model.onnx", "bytes": 541624087,
        "sha256": "6f58d7ffa4c33bf49c8c67ac27bc5265a940a139cb67254477997bad41efc16d",
        "input": "input", "outputs": ["predictions", "embeddings"],
        "wanted": ["predictions", "embeddings"],
    },
}
for _m in MODELS.values():
    replay.InferenceSession.signatures[_m["file"].rsplit("/", 1)[-1]] = (_m["input"], _m["outputs"])

_SPLIT_OUT = ("audio_dir", "dim_reduction_model", "testing")

# bacpipe's functions that only draw a picture file (PNG) with matplotlib, seaborn or plotly. Those libraries are not
# loaded here, and the page draws the same data itself, so these are replaced by a function that does nothing.
# Each is named by the module that calls it. Nothing that computes or writes a result is in this list.
DRAWING_ONLY = (
    ("bacpipe.core.workflows", "plot_comparison"),                                    # overview/comp_fig.png
    ("bacpipe.embedding_evaluation.visualization.visualize", "plot_per_class_results"),   # overview/*.png
    ("bacpipe.embedding_evaluation.visualization.visualize", "plot_overview_results"),    # overview/*.png
    ("bacpipe.embedding_evaluation.visualization.visualize_embeddings", "plot_embeddings_px"),   # a figure bacpipe drops
    ("bacpipe.embedding_evaluation.probing.evaluate_probe", "save_confusion_matrix"),     # probing/confusion_matrix_*.png
    ("bacpipe.embedding_evaluation.probing.evaluate_probe", "plot_classification_results"),   # plots/probe_results_*.png
)


def _no_drawing():
    import importlib

    def nothing(*args, **kwargs):
        return None

    for module, name in DRAWING_ONLY:
        mod = importlib.import_module(module)
        getattr(mod, name)                    # fails loudly if bacpipe renames one
        setattr(mod, name, nothing)


_no_drawing()


def _without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


class Engine:
    def __init__(self, model_base_path):
        self.model_base_path = Path(model_base_path)
        self.kwargs, self.loaders, self.failed, self._after = None, {}, [], {}
        self._embed = self._loader = self._kwargs = self._model = None

    # ------------------------------------------------------------------ settings

    @staticmethod
    def defaults():
        """bacpipe's own config.yaml and settings.yaml, as shipped."""
        return {"config": dict(vars(bacpipe.config)), "settings": dict(vars(bacpipe.settings))}

    def configure(self, config, settings, save_logs_too=True):
        """The first lines of bacpipe.play(): merge the choices with bacpipe's defaults.

        The choices are also set on bacpipe.config and bacpipe.settings, as bacpipe's README does
        (`bacpipe.settings.device = ...`), because parts of bacpipe read them there.
        """
        for namespace, values in ((bacpipe.config, config), (bacpipe.settings, settings)):
            for key, value in values.items():
                setattr(namespace, key, value)
        kwargs = replace_default_kwargs_with_user_kwargs(**config, **settings)
        kwargs["model_base_path"] = str(self.model_base_path)
        for name in kwargs["models"]:                # ensure_models_exist looks for a folder that is not empty
            target = self.model_base_path / MODELS[workflows.confirm_model_name(name)]["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text("The model runs in the page. This file marks it as present for bacpipe.\n")
        kwargs["model_base_path"] = workflows.ensure_models_exist(self.model_base_path, model_names=kwargs.get("models"))
        if save_logs_too:
            save_logs(**kwargs)
        kwargs["models"] = workflows.get_model_names(**kwargs)
        self.kwargs, self.loaders, self.failed, self._after = kwargs, {}, [], {}
        return {"models": kwargs["models"], "dataset": Path(kwargs["audio_dir"]).stem,
                "results": str(Path(kwargs["main_results_dir"]) / Path(kwargs["audio_dir"]).stem)}

    def needs_run(self):
        """play(): `if overwrite or not evaluation_with_settings_already_exists(**kwargs)`."""
        return bool(self.kwargs.get("overwrite")) or not workflows.evaluation_with_settings_already_exists(**self.kwargs)

    # ------------------------------------------------------------------ embeddings, one model

    def start_model(self, model_name):
        """run_pipeline_for_models -> run_pipeline_for_single_model -> generate_embeddings, up to the Embedder."""
        # run_pipeline_for_models passes everything on except `models`
        kwargs = _without(self.kwargs, "models")
        audio_dir, dim_reduction_model = kwargs["audio_dir"], kwargs["dim_reduction_model"]
        testing = kwargs.get("testing", False)
        check_if_already_processed = kwargs.get("check_if_already_processed", False)
        check_if_already_dim_reduced = kwargs.get("check_if_already_dim_reduced", True)
        kwargs = _without(kwargs, *_SPLIT_OUT, "check_if_already_processed", "check_if_already_dim_reduced")

        # run_pipeline_for_single_model
        if dim_reduction_model is None:
            dim_reduction_model = "None"
        model_name = workflows.confirm_model_name(model_name, **kwargs)
        kwargs = replace_default_kwargs_with_user_kwargs(remove_keys=list(_SPLIT_OUT), **kwargs)
        workflows.get_paths = make_set_paths_func(audio_dir, testing=testing, **kwargs)   # its `global get_paths`
        paths = workflows.get_paths(model_name)
        self._after[model_name] = dict(model_name=model_name, audio_dir=audio_dir, dim_reduction_model=dim_reduction_model,
                           check_if_already_dim_reduced=check_if_already_dim_reduced, testing=testing,
                           kwargs=dict(kwargs), paths=paths)

        # generate_embeddings(model_name, audio_dir, check_if_combination_exists=check_if_already_processed,
        #                     paths=paths, testing=testing, **kwargs)
        kwargs = dict(kwargs, check_if_combination_exists=check_if_already_processed, paths=paths, testing=testing)
        workflows.ensure_models_exist(model_names=model_name, model_base_path=kwargs.get("model_base_path"))
        logger.info(f"\n\n\n###### Generating embeddings using {model_name.upper()} ######\n")
        kwargs.pop("avoid_pipelined_gpu_inference", None)
        kwargs = replace_default_kwargs_with_user_kwargs(remove_keys=list(_SPLIT_OUT), **kwargs)
        use_folder_structure = kwargs.pop("use_folder_structure", True)
        loader = Loader(use_folder_structure=use_folder_structure, audio_dir=audio_dir, model_name=model_name, **kwargs)
        self._model, self._loader, self._kwargs, self._embed = model_name, loader, kwargs, None
        self.loaders[model_name] = loader
        spec = MODELS[model_name]
        info = {"model": model_name, "already_computed": bool(loader.combination_already_exists),
                "file": spec["file"], "input": spec["input"], "wanted": spec["wanted"], "files": []}
        if not loader.combination_already_exists:
            self._embed = Embedder(loader=loader, model_name=model_name, **kwargs)
            self._session = replay.InferenceSession.sessions[spec["file"].rsplit("/", 1)[-1]]
            self._files = list(loader.files)                    # the recordings still to do, in bacpipe's order
            info.update(files=[f.relative_to(loader.audio_dir).as_posix() for f in self._files],
                        sample_rate=self._embed.model.sr, segment_length=self._embed.model.segment_length,
                        batch_size=self._embed.model.batch_size,
                        classifier=bool(self._embed.model.bool_classifier),
                        continued=bool(loader.continue_incomplete_run), embed_dir=str(loader.embed_dir))
        return info

    def prepare(self, index):
        """Embedder.run_inference_pipeline_sequentially, first half: the audio windows of one recording.

        Returns a float32 array (windows, samples), or None when the recording could not be read: it is then
        logged and listed in `failed`, as bacpipe logs it and carries on.
        """
        file = self._files[index]
        try:
            self._frames = self._embed.prepare_audio(file)
        except Exception as e:                      # bacpipe: "Error loading audio, skipping file."
            self._skip(file, f"could not be read: {e}")
            return None
        return np.ascontiguousarray(self._frames, dtype=np.float32)

    def store(self, index, outputs):
        """Second half: give the network's outputs to bacpipe, which then does what it does after the model.

        outputs: {output name: float32 array with one row per window}, for the names in `wanted`.
        """
        file, embed, loader = self._files[index], self._embed, self._loader
        try:
            self._session.feed(outputs)
            embeddings = embed.get_embeddings_for_audio(self._frames)
            if self._session.pending_rows():
                raise RuntimeError(f"{self._session.pending_rows()} windows were computed but not used")
        except Exception as e:                      # bacpipe: "Error generating embeddings, skipping file."
            self._skip(file, f"embeddings failed: {e}")
            return 0
        finally:
            self._frames = None
        loader._write_audio_file_to_metadata(file, embed.model, embeddings, embed.file_length)
        loader.save_embedding_file(file, embeddings)
        if embed.model.bool_classifier:
            embed.classifier.save_classifier_outputs(loader, file)
        return int(embeddings.shape[0])

    def store_flat(self, index, names, arrays, n):
        """`store` for the page: each output arrives as one flat float32 buffer for the recording's n windows."""
        outputs = {}
        for name, array in zip(names, arrays):
            data = array.to_py() if hasattr(array, "to_py") else array
            outputs[name] = np.asarray(data, dtype=np.float32).reshape(int(n), -1)
        return self.store(index, outputs)

    def skip(self, index, reason):
        """The page could not run the model on this recording."""
        self._skip(self._files[index], reason)
        self._frames = None

    def _skip(self, file, reason):
        logger.warning(f"\n Skipping {file}: {reason}\n")
        self.failed.append({"model": self._model, "file": Path(file).relative_to(self._loader.audio_dir).as_posix(),
                            "reason": str(reason)})
        if self._embed is not None and getattr(self._embed.model, "bool_classifier", False):
            self._embed.classifier.predictions = torch.tensor([])   # not carried over into the next recording

    def finish_model(self):
        """generate_embeddings, after the loop ("Finalize"), then the rest of run_pipeline_for_single_model."""
        loader, embed, kwargs = self._loader, self._embed, self._kwargs
        if embed is not None:
            if not loader.metadata_dict["files"]["audio_files"]:
                raise FileNotFoundError("No embedding files were generated. Every recording failed: see the log.")
            if embed.model.bool_classifier and not embed.dim_reduction_model:
                try:
                    embed.classifier.save_annotation_table(loader, **kwargs)
                except Exception as e:
                    logger.warning("Error when trying to save classifier predictions. "
                                   f"Continuing but only embeddings will be saved. {str(e)}")
            loader.write_metadata_file()
            loader.update_files()
        self._embed = self._session = self._files = None
        return {"model": self._model, "files": len(loader.metadata_dict["files"]["audio_files"]),
                "embeddings": int(sum(loader.metadata_dict["files"]["nr_embeds_per_file"])),
                "embed_dir": str(loader.embed_dir)}

    def reduce(self, model_name=None):
        """run_pipeline_for_single_model, second half: the dimensionality reduction and the labels for its plot."""
        a = self._after[model_name or self._model]
        kwargs, paths, dim_reduction_model = a["kwargs"], a["paths"], a["dim_reduction_model"]
        if dim_reduction_model in ["None", False, None, ""]:
            return None
        loader_dim_reduced = workflows.generate_embeddings(
            model_name=a["model_name"], dim_reduction_model=dim_reduction_model, audio_dir=a["audio_dir"],
            check_if_combination_exists=a["check_if_already_dim_reduced"], testing=a["testing"], **kwargs)
        if not (paths.plot_path.joinpath("embeddings.png").exists() or a["testing"]):
            vis_loader = EmbedAndLabelLoader(dim_reduction_model=dim_reduction_model, **kwargs)
            try:
                plot_embeddings(vis_loader, paths=paths, model_name=loader_dim_reduced.model_name,
                                dim_reduction_model=dim_reduction_model, bool_plot_centroids=False,
                                label_by="time_of_day", **kwargs)
            except AssertionError as e:
                logger.exception("Plotting of embeddings has failed. Continuing with processing "
                                 f"embeddings, but this will cause evaluation problems later on. {str(e)}")
        return str(loader_dim_reduced.embed_dir)

    # ------------------------------------------------------------------ evaluation, all models

    def evaluate(self):
        """play(): model_specific_evaluation(loader_dict, **kwargs), then cross_model_evaluation(**kwargs)."""
        kwargs = dict(self.kwargs, models=[m for m in self.kwargs["models"] if m in self.loaders])
        t0 = time.time()
        workflows.model_specific_evaluation(self.loaders, **kwargs)
        workflows.cross_model_evaluation(**kwargs)
        return {"seconds": round(time.time() - t0, 1)}
