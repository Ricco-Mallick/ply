
"""Fetch the E57 from Drive inside Modal (server-side), decimate, publish a coloured PLY."""
import modal

app = modal.App("ply-lidar")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("curl")
         .pip_install("numpy", "pye57", "huggingface_hub", "requests"))
DRIVE = "1FahPjNdPSe1sz9vuO3jiUyTc21DplKBv"


@app.function(image=image, cpu=8, memory=32768, timeout=3600, secrets=[modal.Secret.from_name("ply-hf")])
def ingest():
    import os, subprocess, time
    import numpy as np
    import pye57
    from huggingface_hub import HfApi

    path = "/tmp/office.e57"
    t0 = time.time()
    subprocess.run(["curl", "-sL", "-o", path,
                    f"https://drive.usercontent.google.com/download?id={DRIVE}&export=download&confirm=t"], check=True)
    print("downloaded", round(os.path.getsize(path) / 2**20, 1), "MB in", round(time.time() - t0, 1), "s", flush=True)
    with open(path, "rb") as f:
        assert f.read(8) == b"ASTM-E57", "not an E57"

    e = pye57.E57(path)
    d = e.read_scan_raw(0, ignore_unsupported_fields=True)
    n = d["cartesianX"].size
    step = max(1, n // 6_000_000)
    idx = np.arange(0, n, step)
    print("points", n, "keeping", idx.size, flush=True)
    x = d["cartesianX"][idx].astype(np.float32)
    y = d["cartesianY"][idx].astype(np.float32)
    z = d["cartesianZ"][idx].astype(np.float32)
    rgb = np.stack([d["colorRed"][idx], d["colorGreen"][idx], d["colorBlue"][idx]], 1)
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / max(1.0, float(rgb.max())) * 255).astype(np.uint8)
    out = np.zeros(len(idx), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    out["x"], out["y"], out["z"] = x, -z, y
    out["red"], out["green"], out["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    ply = "/tmp/office-1.ply"
    with open(ply, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(out)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        f.write(out.tobytes())
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(path_or_fileobj=ply, path_in_repo="lidar/office-1/office-1.ply",
                    repo_id="Ricco-Marcelino/ply-data", repo_type="dataset")
    bounds = {"x": [float(x.min()), float(x.max())], "y": [float(-z).min(), float(-z).max()], "z": [float(y.min()), float(y.max())]}
    print("published office-1.ply", round(len(out) / 1e6, 2), "M points", bounds, flush=True)
    return {"points": int(len(out)), "bounds": bounds}


@app.local_entrypoint()
def main():
    print(ingest.remote())
