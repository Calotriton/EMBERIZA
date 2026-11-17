# 0001. Does EMBERIZA give bacpipe's results?

Date: 2026-11-17 · Informs: [ADR 0001](../decisions/0001-bacpipe-in-the-browser.md), [ADR 0002](../decisions/0002-engine-and-stand-ins.md), [ADR 0003](../decisions/0003-umap-port.md), [ADR 0006](../decisions/0006-network-batch-size.md) · Tools: [`tests/tools/`](../../tests/tools/), [`tools/bacpipe_reference.py`](../../tools/bacpipe_reference.py)

## Summary

Yes, for everything bacpipe computes without unseeded random numbers. The one exception is the UMAP map, which keeps neighbours as well as bacpipe's but is not the same picture.

- **Outside the browser**, on the same operating system as the reference, EMBERIZA's engine wrote the same embeddings as bacpipe to the last bit (20 of 20 files). `metadata.yml`, every label table and BirdNET's predictions were identical too. Perch's confidence scores were within 0.000003.
- **In the browser** the embeddings were within 0.000005 of bacpipe's, and the confidence scores within 0.000006. The label tables were identical, apart from the Windows bug below. The kNN probe gave bacpipe's scores.
- **k-means and the linear probe** differ from run to run in bacpipe itself, because bacpipe does not seed them.
- **UMAP:** the same graph, and the same update loop to the last bit from the same start. The final positions differ. Trustworthiness was within 0.018 of bacpipe's maps.
- **A bacpipe bug on Windows:** with recordings in sub-folders, bacpipe 1.3.5 puts the "default classifier" label on the wrong windows. EMBERIZA in the browser does not.

## Methods

**Reference.** bacpipe 1.3.5 (commit `d7aa3a2971ae4287c9f4a10acde324519776296a`), run by `tools/bacpipe_reference.py` in bacpipe's own environment on Windows: Python 3.11.18, numpy 1.26.4, SciPy 1.17.1, umap-learn 0.5.12, onnxruntime 1.31.1, PyTorch 2.6.0 (CPU). The script makes the calls `bacpipe.play()` makes, without the dashboard. With `dashboard: False`, bacpipe 1.3.5 stops in `plot_embeddings` with an unbound variable `points`.

**Recordings.**
- *Test:* three recordings of one minute at 32 kHz, from the owner's data, not in the repository. 37 windows for Perch 2.0, 61 for BirdNET+ V3.0.
- *test_data:* bacpipe's own test data. Seven recordings at 48 kHz, mono and stereo, in two sub-folders, with bacpipe's `annotations.csv`. 3 min 1 s of sound, 37 windows for Perch 2.0, 63 for BirdNET+ V3.0.

Both with Perch 2.0 and BirdNET+ V3.0 (preview 3.1), and bacpipe's default settings. test_data also with clustering and probing.

**EMBERIZA, outside the browser.** `tests/tools/run_engine.py` runs `app/py/emberiza/engine.py` with the package versions Pyodide provides (Python 3.14.10, numpy 2.4.6, SciPy 1.18.0, scikit-learn 1.8.0) and the stand-ins of ADR 0002. The real onnxruntime 1.32.0 takes the place of ONNX Runtime Web.

**EMBERIZA, in the browser.** `tests/tools/browser_e2e.py` drives the page in Edge (headless) through Playwright. It chooses the folder, runs, downloads the ZIP and compares it. Pyodide 314.0.7 and ONNX Runtime Web 1.30.0 in WebAssembly, on a laptop processor with 4 threads.

**Comparison.** `tests/tools/compare_results.py` compares the two result folders file by file. It leaves out the time stamps in folder names and the absolute paths, and reads "\" and "/" as the same. Numbers are compared by their largest absolute difference, text exactly. PNG files and the probes' `.pt` files are not compared.

## Results

### Embeddings, labels and predictions

| | Outside the browser, Test | Outside the browser, test_data | Browser, Test | Browser, test_data |
|---|---|---|---|---|
| Embeddings (`.npy`) | identical, 6 of 6 | identical, 14 of 14 | within 4.3e-6 | within 2.7e-6 |
| `metadata.yml` | identical | identical | identical | identical |
| Label tables (`.csv`) | identical | identical, 8 of 8 | identical | identical except `default_classifier` (below) |
| BirdNET+ V3.0 predictions | identical | identical | both models within 5.6e-6 | both models within 1.8e-6 |
| Perch 2.0 predictions | within 3.0e-6 | within 1.7e-6 | | |
| Raven tables, "High Freq (Hz)" | within 0.095 | within 0.053 | within 0.18 | within 0.057 |

- **Perch's predictions outside the browser.** bacpipe applies a softmax over Perch's 14,795 classes in PyTorch. The stand-in computes it in float64 and rounds to float32. Summing that many float32 terms in another order moves the sixth decimal.
- **Raven tables.** bacpipe writes the confidence times the sample rate (32,000) in the column "High Freq (Hz)" (`runner.py`, line 1259). A difference of 0.000003 in confidence is 0.095 there. The other columns were identical, apart from "\" and "/" in the paths.
- **The audio windows** fed to the networks were compared directly in an earlier run and were identical for all 20 pairs of recording and model. The bit-identical embeddings outside the browser confirm it.

### Evaluation (test_data)

| | bacpipe | Outside the browser | Browser |
|---|---|---|---|
| kNN probe, Perch 2.0 (macro accuracy, micro accuracy, AUC) | 0.667, 0.714, 1.000 | the same | the same |
| kNN probe, BirdNET+ V3.0 | 0.750, 0.909, 0.984 | the same | the same |
| Linear probe, Perch 2.0 | 0.667, 0.714, 1.000 | the same | the same |
| Linear probe, BirdNET+ V3.0 | 0.500, 0.727, 0.765 | 0.500, 0.727, 0.781 | 0.500, 0.727, 0.723 |
| Clustering scores without k-means | | identical, 36 of 36 per model | identical, apart from `default_classifier` |
| Clustering scores with k-means | | differ | differ |

bacpipe seeds only the split of the probing data, t-SNE and UMAP. The linear probe's starting weights and k-means start from unseeded random numbers, so they change between two runs of bacpipe itself. The split for probing was identical in all three.

### The default classifier label on Windows

bacpipe's `fill_remaining_labels` adds a "below_thresh" row for every window without a prediction, then sorts all rows by file name (`label_embeddings.py`, line 563). On Windows the predicted rows and the added rows write the same file with different separators ("\" and "/"). The sort then interleaves them out of step with the windows. In the reference, the label was on the wrong window for 28 of 37 Perch windows and 40 of 63 BirdNET windows.

- Outside the browser on Windows, EMBERIZA's engine reproduced this exactly, because it runs bacpipe's code with Windows paths.
- In the browser, paths use "/" as on Linux, and every label was on its window. With the labels in place, the clustering scores against `default_classifier` rose: the AMI of species against default classifier was 0.71 for Perch 2.0 in the browser, against 0.46 in the reference.

### UMAP

From `tests/test_umap.py` (umap-learn's own numbers on 60 made-up points) and the maps of both sets:
- **Graph:** the same edges, with weights, sigmas and rhos within 0.00001.
- **Update loop:** from umap-learn's own start, graph and random state, the port ends at umap-learn's positions to the last bit. The JavaScript loop gives the Python loop's positions to the last bit (checked at each start of the page).
- **End to end:** the positions differ. The start comes from SciPy's `eigsh`, and SciPy differs (1.18 against 1.17). umap-learn seeds each point's random numbers from the bits of its start position, so a start that differs in the last bit takes another path.
- **Trustworthiness** (scikit-learn, 10 neighbours), bacpipe's maps against the port on bacpipe's own embeddings:

| | bacpipe | port | browser |
|---|---|---|---|
| Test, Perch 2.0 | 0.899 | 0.881 | |
| Test, BirdNET+ V3.0 | 0.864 | 0.881 | |
| test_data, Perch 2.0 | 0.970 | 0.975 | 0.968 |
| test_data, BirdNET+ V3.0 | 0.902 | 0.892 | 0.892 |

### Speed and start-up (browser)

- The page was ready after 15 to 25 s, with nothing cached, in the faster runs.
- Test, both models: 37 s for the whole run. Perch 2.0 embedded 4.6 windows a second and BirdNET+ V3.0 3.6.
- test_data, both models, with clustering and probing: 33 s. 4.4 and 4.2 windows a second.
- The same test_data run, repeated twice later on the same laptop, took 99 s and 186 s (Perch 2.0 down to 0.8 windows a second), and the page took up to 57 s to start. The laptop was on battery during the slowest run. Its power state during the others was not recorded. Speed in the browser varies with the state of the machine, so these numbers are a range, not a promise.

## Conclusions

- The engine follows bacpipe faithfully: on the same system it is bacpipe, file for file, except for numbers that bacpipe itself does not fix.
- The browser's own arithmetic moves the embeddings in the sixth decimal, which no label, probe or clustering result picked up on these sets.
- UMAP maps should be read as equivalent, not identical. The README and the About page say so.
- The Windows label bug is bacpipe's. It may be worth reporting to bacpipe's authors.
