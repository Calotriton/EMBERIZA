# Licences of what EMBERIZA contains and uses

EMBERIZA's own code is under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

## Included in this repository

| What | Where | Licence |
|---|---|---|
| bacpipe 1.3.5, a subset, unchanged | `app/py/bacpipe/` | Apache 2.0, in `app/py/bacpipe/LICENSE` |
| A port of parts of umap-learn 0.5.12 | `app/py/emberiza/umap_port.py` | BSD 3-Clause, below |
| coi-serviceworker 0.1.7 | `app/coi-serviceworker.min.js` | MIT, notice in the file |
| Class lists of the two models | `app/py/bacpipe/model_pipelines/model_specific_utils/` | as bacpipe distributes them |

## Downloaded by the browser when the page is used

Nothing below is stored in this repository. The page fetches it from the place named.

| What | From | Licence |
|---|---|---|
| Perch 2.0 as ONNX, `perch_v2_no_dft.onnx` | bacpipe's model repository on Hugging Face, `vskode/bacpipe_models` | Apache 2.0. Model © Google. Conversion by Justin Chu. |
| BirdNET+ V3.0 developer preview 3.1 as ONNX, `model.onnx` | the same repository | CC BY-SA 4.0 with terms of use, below |
| Pyodide 314.0.7 with numpy, scipy, pandas, scikit-learn, soundfile, soxr, pyyaml | jsDelivr | MPL 2.0 (Pyodide), BSD (numpy, scipy, pandas, scikit-learn, soundfile), LGPL 2.1 (libsndfile, libsoxr), MIT (pyyaml) |
| ONNX Runtime Web 1.30.0 | jsDelivr | MIT |
| Plotly.js 2.35.2 | jsDelivr | MIT |
| fflate 0.8.3 | jsDelivr | MIT |

## BirdNET+ V3.0 developer preview: terms of use

The model is published by the BirdNET team at [doi:10.5281/zenodo.20703646](https://doi.org/10.5281/zenodo.20703646)
under CC BY-SA 4.0, with terms of use that add two prohibitions. In the words of that file, the models must not be
used:

1. for poaching, or wildlife tracking intended to facilitate illegal capture, harm, trade or exploitation of species;
2. for any military purpose, including surveillance, targeting or integration into defense systems.

Tools and publications that use the model must attribute it, by citation or by an acknowledgment such as
"Powered by BirdNET". EMBERIZA shows this in its About section. The model is a developer preview: its authors
say that models and labels will change, and that it lacks a noise class.

## Perch 2.0

Perch 2.0 © Google, Apache 2.0. Cite van Merriënboer et al. (2025), *Perch 2.0: The Bittern Lesson for Bioacoustics*,
[arXiv:2508.04665](https://arxiv.org/abs/2508.04665). The licence grants no trademark rights. EMBERIZA is not
affiliated with Google.

## umap-learn

```
BSD 3-Clause License

Copyright (c) 2017, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
