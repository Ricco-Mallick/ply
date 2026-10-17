"""Results storage on the Hugging Face Hub (a private dataset repo) instead of the bucket.

Why: the bucket's free tier counts every download against 1 GB a day and caps storage at 10 GB, while a private
HF repo is 100 GB free with no daily download limit (and its links answer range requests, so the API can proxy
them like it copies from the bucket). Enabled by setting PLY_HF_REPO=<owner>/<repo> and HF_TOKEN with write
access; without them every call raises Missing and the caller falls back to the bucket, so nothing breaks.

Layout mirrors the bucket: scenes/<jobId>/<file> -> <repo>/scenes/<jobId>/<file>.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


class Missing(Exception):
    """No HF configuration, or the object is not on HF: the caller should use the bucket."""


def configured() -> bool:
    return bool(os.environ.get("PLY_HF_REPO") and os.environ.get("HF_TOKEN"))


def _api():
    if not configured():
        raise Missing("PLY_HF_REPO / HF_TOKEN not set")
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # the Kaggle fast-lane image has no huggingface_hub
        raise Missing(f"huggingface_hub not installed: {e}") from e
    return HfApi(token=os.environ["HF_TOKEN"])


def upload(key: str, path: str | Path) -> None:
    api = _api()
    api.upload_file(path_or_fileobj=str(path), path_in_repo=key, repo_id=os.environ["PLY_HF_REPO"], repo_type="dataset")


def download(key: str, dest: str | Path) -> None:
    from huggingface_hub import hf_hub_download

    _api()  # checks configuration
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = hf_hub_download(repo_id=os.environ["PLY_HF_REPO"], repo_type="dataset", filename=key, token=os.environ["HF_TOKEN"], local_dir=str(dest.parent))
    except Exception as e:  # EntryNotFoundError, a network hiccup, ...
        raise Missing(str(e)) from e
    if Path(p) != dest:
        shutil.copyfile(p, dest)


def url(key: str) -> str:
    """A resolve URL the API can fetch with the same bearer token (supports range requests)."""
    if not configured():
        raise Missing("PLY_HF_REPO / HF_TOKEN not set")
    return f"https://huggingface.co/datasets/{os.environ['PLY_HF_REPO']}/resolve/main/{key}"
