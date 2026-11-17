# 0004. The model files are bacpipe's own, from its Hugging Face repository at a fixed commit
Status: Accepted
Date: 2026-11-16

## Context

The page needs the ONNX files of Perch 2.0 (413 MB) and BirdNET+ V3.0 (542 MB). GitHub Pages limits a site to 1 GB and a file to 100 MB, so the files cannot be part of the site. bacpipe downloads the same files from its own Hugging Face dataset, `vskode/bacpipe_models`, into `model_base_path`.

The request was the latest BirdNET bacpipe supports. In bacpipe 1.3.5 that is `birdnet_v3`, BirdNET+ V3.0 developer preview 3.1.

## Options considered

1. **bacpipe's own files, from its repository, pinned to one commit.**
   - Gains: the networks are exactly the ones bacpipe runs, so parity holds by construction. Nothing is copied or redistributed by EMBERIZA. Hugging Face serves the files with the CORS headers a page needs.
   - Loses: the page depends on bacpipe's repository staying online at that commit.
2. **A copy in a repository of EMBERIZA's.**
   - Gains: independence from bacpipe's repository.
   - Loses: EMBERIZA would redistribute the models, and must then carry their licences and terms as a distributor. Two copies can drift apart.
3. **Convert the models anew** (from Google's and BirdNET's releases).
   - Loses: another conversion is another network in its last digits, and parity with bacpipe is lost.

## Decision

Option 1.
- **Commit:** `f10c46a68eb9fcb18abe89394893400988248a70` of `vskode/bacpipe_models`.
- **Files:** `perch_v2/perch_v2_no_dft.onnx` and `birdnet_v3/model.onnx`. Their sizes and SHA-256 are in `app/py/emberiza/engine.py`. The worker checks both before it uses a file, and refuses a file that does not match.
- **Cache:** the browser keeps each file after its first download (Cache Storage), so a model downloads once.
- **Development:** on `127.0.0.1` the page reads the files from `model/`, where `tools/bacpipe_reference.py` puts them through bacpipe's own `ensure_models_exist`.
- **Licences and terms:** the page names each model's licence before the first download. It shows BirdNET's terms of use (no poaching, no illegal harm or trade of wildlife, no military use) and "Powered by BirdNET".

## Evidence

- The SHA-256 of both files matched in every run of the browser test.
- The embeddings made from these files match bacpipe's ([Research 0001](../research/0001-parity-with-bacpipe.md)).

## Consequences

- **Easier:** no hosting of large files, and no redistribution of the models.
- **Harder:** a change of bacpipe's repository needs a new commit and new hashes here.
- **Reversal:** `MODEL_COMMIT` and `MODEL_BASE` in `app/worker.js` point elsewhere, and the hashes stay the guard.
