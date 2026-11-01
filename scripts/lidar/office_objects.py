
"""Furnish the office: cluster the LiDAR interior into objects and write them into shell.json."""
import modal
app = modal.App("ply-lidar-objects")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "huggingface_hub")

@app.function(image=image, cpu=8, memory=24576, timeout=2400, secrets=[modal.Secret.from_name("ply-hf")])
def build():
    import json, os, re
    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download
    repo = "Ricco-Marcelino/ply-data"; tok = os.environ["HF_TOKEN"]
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename="lidar/office-1/office-1.ply", token=tok)
    sp = hf_hub_download(repo_id=repo, repo_type="dataset", filename="scenes/lidar-office-1/shell.json", token=tok)
    shell = json.load(open(sp))

    hdr = b""
    with open(p, "rb") as f:
        while not hdr.endswith(b"end_header\n"):
            hdr += f.read(1)
        off = len(hdr)
    raw = [(t.decode(), nm.decode()) for t, nm in re.findall(rb"property (\w+) (\w+)", hdr)]
    fmt = {"float": "<f4", "uchar": "u1", "double": "<f8"}
    n = int(re.search(rb"element vertex (\d+)", hdr).group(1))
    dt = np.dtype([(nm, fmt.get(t, "<f4")) for t, nm in raw])
    a = np.memmap(p, dtype=dt, mode="r", offset=off, shape=(n,))
    names = [nm for _, nm in raw]
    P = np.stack([a[names[0]], a[names[1]], a[names[2]]], 1).astype(np.float32)
    floor = float(np.percentile(P[:, 1], 0.5)); P[:, 1] -= floor

    # keep the interior: away from the walls, above the floor, below the ceiling
    keep = np.ones(len(P), bool)
    for w in shell.get("walls", []):
        ax, az = w["a"]; bx, bz = w["b"]
        dx, dz = bx - ax, bz - az; L2 = dx*dx + dz*dz or 1
        t = np.clip(((P[:, 0]-ax)*dx + (P[:, 2]-az)*dz) / L2, 0, 1)
        d = np.hypot(P[:, 0]-(ax+dx*t), P[:, 2]-(az+dz*t))
        keep &= d > 0.18
    keep &= (P[:, 1] > 0.12) & (P[:, 1] < min(2.5, shell["ceilingY"] - 0.4))
    Q = P[keep]
    print("interior points", len(Q), flush=True)

    # furniture per cell: connected components merged most of the room into one blob, so tile the floor
    # plan into 1.0 m cells and measure what stands in each (a partition spanning several cells becomes a few)
    cell = 1.0
    objects = []
    x0, z0 = Q[:, 0].min(), Q[:, 2].min()
    ix = ((Q[:, 0] - x0) // cell).astype(np.int64)
    iz = ((Q[:, 2] - z0) // cell).astype(np.int64)
    key = ix * 10000 + iz
    for k in np.unique(key):
        pts = Q[key == k]
        if len(pts) < 350: continue
        lo = np.percentile(pts, 2, axis=0); hi = np.percentile(pts, 98, axis=0)
        size = hi - lo
        if size[1] < 0.25: continue
        centre = (lo + hi) / 2
        h = float(size[1]); w = float(size[0]); d = float(size[2])
        if h < 0.5: kind = "low table"
        elif h < 0.85 and max(w, d) > 0.9: kind = "desk"
        elif h < 1.0: kind = "chair"
        elif h < 1.5 and min(w, d) < 0.4: kind = "partition"
        elif h >= 1.3: kind = "cabinet"
        else: kind = "desk"
        objects.append({"id": f"{kind.replace(' ','-')}-{len(objects)}", "name": kind.title(), "kind": kind,
                        "center": [round(float(c), 3) for c in centre],
                        "half": [round(w/2, 3), round(h/2, 3), round(d/2, 3)], "yaw": 0.0, "points": int(len(pts))})
    objects.sort(key=lambda o: -o["points"])
    objects = objects[:220]
    shell["objects"] = objects
    counts = {}
    for o in objects: counts[o["kind"]] = counts.get(o["kind"], 0) + 1
    print("objects", len(objects), counts, flush=True)
    fp = "/tmp/shell.json"; open(fp, "w").write(json.dumps(shell, indent=1))
    HfApi(token=tok).upload_file(path_or_fileobj=fp, path_in_repo="scenes/lidar-office-1/shell.json", repo_id=repo, repo_type="dataset")
    return {"objects": len(objects), "kinds": counts}

@app.local_entrypoint()
def main():
    print(build.remote())
