# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The GPU worker's prebuilt CUDA wheels, kept on Hugging Face.

The `wheels` workflow builds gsplat (at the pinned commit), fused-ssim, fused-bilagrid and ppisp, but its artifacts
are deleted after 30 days, and without them no Modal image builds. This keeps a copy under `wheels/` in the results
dataset and brings it back into pipeline/wheels, where pipeline/modal/ply_modal.py reads it. Both commands go through
the Modal `ply-hf` secret, so no Hugging Face token is needed on the machine that runs them.

    modal run scripts/wheels.py::fetch      # pipeline/wheels.lock.json -> pipeline/wheels/*.whl, checksums verified
    modal run scripts/wheels.py::publish    # pipeline/wheels/*.whl -> the dataset; rewrites the lock file

After a rebuild: `gh run download <run> -D /tmp/wh`, copy the .whl files flat into pipeline/wheels, publish with a
`--tag` that names the build (Python, CUDA toolkit, TORCH_CUDA_ARCH_LIST), and commit the lock file.
"""
import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
WHEELS = ROOT / "pipeline" / "wheels"
LOCK = ROOT / "pipeline" / "wheels.lock.json"
REPO = "Ricco-Marcelino/ply-data"

app = modal.App("ply-wheels")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")
hf = modal.Secret.from_name("ply-hf")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.function(image=image, timeout=900, secrets=[hf])
def upload(files: dict[str, bytes], prefix: str) -> str:
    """Write the files and a SHA256SUMS under <prefix>/ in one commit. Returns the commit id."""
    import os
    import tempfile

    from huggingface_hub import HfApi

    d = Path(tempfile.mkdtemp())
    for name, data in files.items():
        (d / name).write_bytes(data)
    (d / "SHA256SUMS").write_text("".join(f"{sha256(data)}  {name}\n" for name, data in sorted(files.items())))
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(folder_path=str(d), path_in_repo=prefix, repo_id=REPO, repo_type="dataset",
                      commit_message=f"wheels: {prefix}")
    # the dataset's head, not the upload's return value: an unchanged re-upload makes no commit
    return api.dataset_info(REPO).sha


@app.function(image=image, timeout=900, secrets=[hf])
def download(prefix: str, revision: str, names: list[str]) -> dict[str, bytes]:
    import os

    from huggingface_hub import hf_hub_download

    return {n: Path(hf_hub_download(REPO, f"{prefix}/{n}", repo_type="dataset", revision=revision,
                                    token=os.environ["HF_TOKEN"])).read_bytes() for n in names}


@app.local_entrypoint()
def publish(tag: str = "cp312-cu128-sm89"):
    files = {p.name: p.read_bytes() for p in sorted(WHEELS.glob("*.whl"))}
    if not files:
        raise SystemExit(f"no .whl files in {WHEELS}")
    prefix = f"wheels/{tag}"
    revision = upload.remote(files, prefix)
    lock = {"repo": REPO, "prefix": prefix, "revision": revision,
            "files": {n: {"sha256": sha256(d), "bytes": len(d)} for n, d in files.items()}}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"published {len(files)} wheels to {REPO}/{prefix} at {revision[:12]}; commit {LOCK.relative_to(ROOT)}")


@app.local_entrypoint()
def fetch():
    lock = json.loads(LOCK.read_text())
    want = lock["files"]
    missing = [n for n, f in want.items()
               if not (WHEELS / n).exists() or sha256((WHEELS / n).read_bytes()) != f["sha256"]]
    if not missing:
        print(f"{WHEELS.relative_to(ROOT)} already matches {LOCK.name}")
        return
    got = download.remote(lock["prefix"], lock["revision"], missing)
    bad = [n for n in missing if sha256(got[n]) != want[n]["sha256"]]
    if bad:
        raise SystemExit(f"checksum mismatch against {LOCK.name}, nothing written: {', '.join(bad)}")
    WHEELS.mkdir(parents=True, exist_ok=True)
    for n in missing:
        (WHEELS / n).write_bytes(got[n])
    print(f"fetched {len(missing)} wheels into {WHEELS.relative_to(ROOT)} "
          f"({lock['prefix']} at {lock['revision'][:12]})")
