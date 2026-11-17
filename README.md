# EMBERIZA

**E**mbedding **M**aps for **B**acpipe: **E**xplore **R**ecordings **I**n a **Z**ero-install **A**pp. *Emberiza* is the genus of the buntings.

EMBERIZA is a browser interface for [bacpipe](https://github.com/bioacoustic-ai/bacpipe), the bioacoustic pipeline by Kather, Haupert, Ghani and Stowell. It runs bacpipe's own code in a browser tab. It computes embeddings of your recordings with Perch 2.0 and BirdNET+ V3.0, then maps, clusters and probes them. There is nothing to install and nothing is uploaded.

EMBERIZA is an independent project. It is not part of bacpipe and is not endorsed by its authors.

## Using it

Open **<https://calotriton.github.io/EMBERIZA/>** in Chrome, Edge or Firefox. The first visit downloads Python and its packages (about 28 MB). Each model downloads the first time it is used: Perch 2.0 is 413 MB and BirdNET+ V3.0 is 542 MB. The browser keeps them.

To run it from your own computer instead, start the local server with `python tools/serve.py` and open <http://127.0.0.1:8000/app/>. The local server reads the model files from `model/` (see Developing).

In the page:

1. **Recordings.** Drop a folder, or choose it. The folder is the dataset and its sub-folders are kept. WAV, FLAC, OGG and MP3 are read. A file `annotations.csv` in the top folder is used as bacpipe uses it.
2. **Models.** Tick Perch 2.0, BirdNET+ V3.0 or both.
3. **Pipeline.** These are the options of bacpipe's `config.yaml`: the method for the map (UMAP, t-SNE, PCA, sparse PCA or none) and the evaluation tasks (clustering, probing).
4. **Settings.** These are the options of bacpipe's `settings.yaml`, in the same groups, with bacpipe's defaults. The **?** next to an option explains it. Options that cannot have an effect in a browser are shown greyed, with the reason. Your choices are remembered, and can be saved as YAML.

Then press **Run bacpipe**. A stopped run carries on where it stopped. The results open in tabs:

- **Explore:** the map of one model's embeddings, coloured by any label bacpipe makes (time of day, date, folder, recording, the model's own classifier, your annotations, the clusters). Click a point to hear that window and see its spectrogram. Select points with the lasso to save them as a table.
- **Compare models:** the maps of both models side by side. A selection in one map is shown in the other.
- **Predictions:** what the model's own classifier found, by recording, hour, day, week or month.
- **Evaluation:** the clustering scores against each label, and the accuracy of the probes per class.
- **Files:** every file bacpipe wrote, and one ZIP of them.
- **Log:** bacpipe's own log.

The results stay in the browser and open again on the next visit. The ZIP holds bacpipe's own folder layout, `bacpipe_results/<dataset>/`, so bacpipe can open it with `already_computed: True`. A results folder made by bacpipe on another computer opens in EMBERIZA the same way, with "Open a bacpipe results folder".

## How it works

- **bacpipe itself runs.** `app/py/bacpipe/` holds bacpipe 1.3.5's files, copied unchanged from one commit. They run in [Pyodide](https://pyodide.org), which is Python compiled for the browser. `app/py/emberiza/engine.py` makes the calls `bacpipe.play()` makes, in the same order.
- **The networks run in ONNX Runtime Web.** Python in a browser cannot wait for the network, so the loop over recordings is cut in two. bacpipe prepares the audio windows of a recording. The page runs the network on them. bacpipe then carries on with the results. The model files are the ones bacpipe downloads, from bacpipe's own repository on Hugging Face, checked by their SHA-256.
- **Stand-ins replace what a browser cannot load.** PyTorch, librosa, audioread and tqdm are replaced by small modules in `app/py/shims/` that do what bacpipe needs from them. The audio one uses the libraries librosa itself calls, soundfile and soxr.
- **UMAP is a port.** umap-learn needs numba, which has no build for the browser. `app/py/emberiza/umap_port.py` follows umap-learn's source step by step. Its update loop also exists in JavaScript, for speed.
- **The page draws.** bacpipe's dashboard needs a Python server, so the page draws the plots itself from bacpipe's result files. bacpipe's picture files (PNG) are not made.

The decisions behind this are in [docs/decisions/](docs/decisions/).

## How close to bacpipe

Measured against bacpipe 1.3.5 on two sets of recordings: three one-minute recordings at 32 kHz, and bacpipe's own test data (seven recordings at 48 kHz, mono and stereo, in two folders, with annotations). The details are in [docs/research/0001-parity-with-bacpipe.md](docs/research/0001-parity-with-bacpipe.md).

| What | Compared with bacpipe |
|---|---|
| Audio windows fed to the models | identical, all 20 recording and model pairs |
| Embeddings, with ONNX Runtime on a processor | identical, all 20 files |
| Embeddings, in the browser | within 0.000005 |
| `metadata.yml`, labels, ground truth, the split for probing | identical |
| BirdNET's predictions | identical outside the browser, within 0.000006 in it |
| Perch's predictions | within 0.000006 |
| kNN probe, and every clustering score without k-means | identical |
| k-means and the linear probe | vary from run to run in bacpipe too, because bacpipe does not seed them |
| PCA map | the same method, scikit-learn's |
| UMAP map | the same neighbour graph, other positions (see below) |
| Picture files (PNG) | not made |

bacpipe's Raven tables write the confidence times the sample rate as "High Freq (Hz)". There the same small differences read as up to 0.2 Hz.

**UMAP maps are not the same picture as bacpipe's.** The port builds the graph umap-learn builds: the same edges, with weights within 0.00001. From the same start, its update loop ends at umap-learn's positions to the last bit. But the start is computed by SciPy's `eigsh`, and Pyodide has SciPy 1.18 where bacpipe's environment had 1.17. The graph's weights also differ in their last digits. umap-learn seeds each point's random numbers from the bits of its start position, so a start that differs in the last bit takes another path. The maps keep neighbours as well as bacpipe's: trustworthiness was within 0.018 of bacpipe's maps on both sets.

## Limits

- **Two models.** bacpipe's other 25 models need PyTorch or TensorFlow.
- **Memory.** A whole recording is read at once, as in bacpipe. Recordings of some hundreds of megabytes may not fit in a browser tab.
- **Speed.** On a laptop processor with 4 threads, Perch 2.0 embedded up to about 4.5 windows a second and BirdNET+ V3.0 up to about 4. Three minutes of sound took 33 s with both models at best, and up to 186 s with the laptop on battery.
- **The graphics card** (WebGPU) can be switched on in Settings. It is experimental and was not validated.

## A bacpipe bug on Windows

With recordings in sub-folders, bacpipe 1.3.5 on Windows puts the "default classifier" label on the wrong windows. It sorts paths written with "\" and with "/" together, so the labels fall out of step with the windows. In bacpipe's own test data this moved the label for 28 of 37 Perch windows. EMBERIZA writes paths with "/", as on Linux, and its labels line up with the windows. The research note has the details.

## Developing

- **Environment:** `uv sync` makes `.venv` with the package versions Pyodide provides. No PyTorch is needed.
- **Tests:** `uv run pytest`. With `EMBERIZA_AUDIO` set to a folder of recordings, `tests/test_parity.py` also compares the engine with a bacpipe reference.
- **A bacpipe reference:** `tools/bacpipe_reference.py <audio folder>`, run with bacpipe's own environment. It also downloads the model files into `model/`.
- **The real page:** `tests/tools/browser_e2e.py <audio folder> <reference folder>` drives it in Edge and compares the ZIP with the reference.
- **Updating bacpipe:** `tools/vendor_bacpipe.py <clone of bacpipe> [commit]`, then `tools/make_manifest.py`, then the tests.
- **The logo:** a male reed bunting in pixel art. Its pixels are text maps in `tools/make_logo.py`, which writes `app/icons/logo.svg` and `app/icons/favicon.svg`.

## Credits

- **bacpipe** by Vincent S. Kather, Sylvain Haupert, Burooj Ghani and Dan Stowell, Apache 2.0. Kather et al. (2026), *Methods in Ecology and Evolution*, [doi:10.1111/2041-210X.70406](https://doi.org/10.1111/2041-210X.70406).
- **Perch 2.0** © Google, Apache 2.0. van Merriënboer et al. (2025), [arXiv:2508.04665](https://arxiv.org/abs/2508.04665). ONNX conversion by Justin Chu.
- **BirdNET+ V3.0 developer preview** by the BirdNET team, [doi:10.5281/zenodo.20703646](https://doi.org/10.5281/zenodo.20703646), CC BY-SA 4.0 with terms of use. Powered by BirdNET.
- **UMAP** after umap-learn by Leland McInnes and contributors, BSD 3-Clause. McInnes, Healy and Melville (2018), [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).
- Built on Pyodide, ONNX Runtime Web, scikit-learn, SciPy, NumPy, pandas, libsndfile, soxr, Plotly.js, fflate and coi-serviceworker.

All licences are in [LICENSES.md](LICENSES.md).

## Citing

If you use EMBERIZA, cite it as given in [CITATION.cff](CITATION.cff) ("Cite this repository" on GitHub). The results are bacpipe's, so cite bacpipe and the models you ran too.

## Licence

EMBERIZA is released under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)), the licence of bacpipe, whose files it includes. The models keep their own licences.
