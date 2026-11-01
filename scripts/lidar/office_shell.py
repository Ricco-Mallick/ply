
"""Measure an office from its LiDAR cloud: floor, ceiling and wall outline, written as shell.json."""
import modal
app = modal.App("ply-lidar-office")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "huggingface_hub")

@app.function(image=image, cpu=4, memory=8192, timeout=1800, secrets=[modal.Secret.from_name("ply-hf")])
def build():
    import io, json, os
    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download
    repo = "Ricco-Marcelino/ply-data"
    tok = os.environ["HF_TOKEN"]
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename="lidar/office-1/office-1.ply", token=tok)

    hdr = b""
    with open(p, "rb") as f:
        while not hdr.endswith(b"end_header\n"):
            hdr += f.read(1)
        off = len(hdr)
    import re
    import re as _re
    raw = [(t.decode(), nm.decode()) for t, nm in _re.findall(rb"property (\w+) (\w+)", hdr)]
    fmt = {"float": "<f4", "uchar": "u1", "double": "<f8"}
    names = [nm for _, nm in raw]
    dt = np.dtype([(nm, fmt.get(t, "<f4")) for t, nm in raw])
    n = int(_re.search(rb"element vertex (\d+)", hdr).group(1))
    a = np.memmap(p, dtype=dt, mode="r", offset=off, shape=(n,))
    x = a[names[0]].astype(np.float32); y = a[names[1]].astype(np.float32); z = a[names[2]].astype(np.float32)
    y = y - np.percentile(y, 0.5)                     # floor at 0
    ceil = float(np.percentile(y, 99.5))
    band = (y > 0.5) & (y < min(2.6, ceil - 0.3))
    pts = np.stack([x[band], z[band]], 1)
    lo = np.percentile(pts, 0.5, axis=0); hi = np.percentile(pts, 99.5, axis=0)
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]

    def hull(P):                                       # monotone chain
        P = np.unique(np.round(P, 3), axis=0)
        P = P[np.lexsort((P[:, 1], P[:, 0]))]
        def cross(o, a_, b): return (a_[0]-o[0])*(b[1]-o[1]) - (a_[1]-o[1])*(b[0]-o[0])
        low, up = [], []
        for q in P:
            while len(low) >= 2 and cross(low[-2], low[-1], q) <= 0: low.pop()
            low.append(q)
        for q in P[::-1]:
            while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0: up.pop()
            up.append(q)
        return np.array(low[:-1] + up[:-1])

    H = hull(pts)
    turns = []
    for i in range(len(H)):
        a0, b0, c0 = H[i-1], H[i], H[(i+1) % len(H)]
        d1, d2 = b0-a0, c0-b0
        cos = float(np.clip(d1 @ d2 / ((np.linalg.norm(d1)*np.linalg.norm(d2)) or 1), -1, 1))
        turns.append(np.degrees(np.arccos(cos)))
    corners = H[[i for i, t in enumerate(turns) if t > 9]] if sum(t > 9 for t in turns) >= 3 else H
    walls = []
    c = pts.mean(axis=0)
    for i in range(len(corners)):
        a0, b0 = corners[i], corners[(i+1) % len(corners)]
        L = float(np.linalg.norm(b0-a0))
        if L < 0.6: continue
        d = (b0-a0)/L; nrm = np.array([-d[1], d[0]]); mid = (a0+b0)/2
        if float(nrm @ (c-mid)) < 0: nrm = -nrm
        walls.append({"a": [round(float(a0[0]),3), round(float(a0[1]),3)], "b": [round(float(b0[0]),3), round(float(b0[1]),3)],
                      "length": round(L,3), "normal": [round(float(nrm[0]),3), round(float(nrm[1]),3)],
                      "angle": round(float(np.degrees(np.arctan2(b0[1]-a0[1], b0[0]-a0[0]))),1)})
    walls.sort(key=lambda w: -w["length"])
    shell = {"version": 1, "frame": "scene, metres, +Y up, floor at 0", "floorY": 0.0, "ceilingY": round(ceil,2),
             "walls": walls, "openings": [], "objects": [], "source": "lidar office-1"}
    fp = "/tmp/shell.json"
    open(fp, "w").write(json.dumps(shell, indent=1))
    api = HfApi(token=tok)
    api.upload_file(path_or_fileobj=fp, path_in_repo="scenes/lidar-office-1/shell.json", repo_id=repo, repo_type="dataset")
    print("walls", len(walls), [(w["length"]) for w in walls], "ceiling", round(ceil,2), flush=True)
    return {"walls": len(walls), "ceiling": ceil}

@app.local_entrypoint()
def main():
    print(build.remote())
