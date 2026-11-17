"""Copies the bacpipe source files that EMBERIZA runs into app/py/bacpipe/, unchanged, from one upstream commit.

    python tools/vendor_bacpipe.py <path to a clone of github.com/bioacoustic-ai/bacpipe> [commit]

Each file is taken from git's object store (the committed bytes, not the working tree), so local edits and line-ending
settings in the clone do not matter. app/py/bacpipe/VENDORED.json records the commit and the SHA-256 of every file.
tests/test_vendored.py checks that nothing in the folder was edited afterwards.

Run tools/make_manifest.py afterwards: the page loads the Python files it lists.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "app" / "py" / "bacpipe"
COMMIT = "d7aa3a2971ae4287c9f4a10acde324519776296a"   # bacpipe 1.3.5, upstream main on 2026-11-09

# Everything bacpipe needs for Perch 2.0 and BirdNET V3.0, its evaluation and its data loading. Left out: the other
# 25 models, the tests, the example notebooks, the type stubs and the logo images.
KEEP = (
    "bacpipe/__init__.py", "bacpipe/config.yaml", "bacpipe/settings.yaml",
    "bacpipe/core/", "bacpipe/embedding_evaluation/", "bacpipe/imgs/__init__.py",
    "bacpipe/model_pipelines/__init__.py", "bacpipe/model_pipelines/model_utils.py", "bacpipe/model_pipelines/runner.py",
    "bacpipe/model_pipelines/dimensionality_reduction/",
    "bacpipe/model_pipelines/feature_extractors/__init__.py",
    "bacpipe/model_pipelines/feature_extractors/perch_v2.py",
    "bacpipe/model_pipelines/feature_extractors/birdnet_v3.py",
    "bacpipe/model_pipelines/model_specific_utils/perch_v2/labels.txt",
    "bacpipe/model_pipelines/model_specific_utils/birdnet_v3/",
)
SKIP_SUFFIXES = (".pyi", ".png", ".md")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def main():
    repo = Path(sys.argv[1])
    commit = sys.argv[2] if len(sys.argv) > 2 else COMMIT
    listed = git(repo, "ls-tree", "-r", "--name-only", "-z", commit, "--", "bacpipe", "LICENSE").decode().split("\0")
    wanted = [p for p in listed if p and not p.endswith(SKIP_SUFFIXES)
              and (p == "LICENSE" or any(p == k or (k.endswith("/") and p.startswith(k)) for k in KEEP))]
    if DEST.exists():
        for old in sorted(DEST.rglob("*"), reverse=True):
            old.unlink() if old.is_file() else old.rmdir()
    files = {}
    for path in wanted:
        data = git(repo, "cat-file", "blob", f"{commit}:{path}")
        rel = "LICENSE" if path == "LICENSE" else path.removeprefix("bacpipe/")
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    date = git(repo, "show", "-s", "--format=%cs", commit).decode().strip()
    (DEST / "VENDORED.json").write_text(json.dumps({
        "source": "https://github.com/bioacoustic-ai/bacpipe", "commit": commit, "commit_date": date,
        "version": "1.3.5", "licence": "Apache-2.0", "modified": False, "files": files}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(files)} files from bacpipe {commit[:7]} ({date}), {sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
