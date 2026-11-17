"""huggingface_hub, for bacpipe's import. bacpipe downloads model files with it. Here the page downloads them."""


def hf_hub_download(repo_id=None, filename=None, **_):
    raise RuntimeError(f"{filename} has to be downloaded by the page, not from Python")
