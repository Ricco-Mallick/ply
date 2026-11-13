
"""Phase 6, step 0: what is in the E57 (scans, scanner poses, fields, images2D), and every scan as one thinned point cloud.

    modal run --detach scripts/lidar/inspect_e57.py --step inspect          # prints one JSON, also lidar/office-1/inspect.json
    modal run --detach scripts/lidar/inspect_e57.py --step export           # lidar/office-1/points-1cm.npz

Both steps start one CPU function with .spawn() and return: read the result from `modal app logs ply-lidar-inspect` or
`modal volume get ply-bench lidar/office-1/inspect.json -`. The E57 is fetched from Drive once and kept on the ply-bench
Volume (lidar/office-1/office-1.e57); the second step and `plypipe lidar` never download it again. The timeouts are the
stop-loss (docs/MISTAKES.md): a function that runs into one is killed and bills its whole timeout at most.

`inspect` reads headers and the first 200k points of each scan (never a whole scan): 2 CPUs, 8 GB, 1800 s. The download
and the header read take minutes; about $0.02.

`export_points` streams every scan in 2M-point chunks into a hash of 1 cm voxels (the mean position and colour of what
falls in each), so the 184.6 M points are never in memory at once, only the voxels: at most ~30 M of them at 56 bytes,
about 2 GB, three times that while a chunk is merged in. 4 CPUs, 16 GB, 3600 s; the merge sorts, which is the cost.
Expected 5 to 15 minutes and under $0.15; the timeout caps it at $0.35.
"""
import json
import os
import time

import modal

app = modal.App("ply-lidar-inspect")
vol = modal.Volume.from_name("ply-bench", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("curl")
         .pip_install("numpy", "pye57"))
DRIVE = "1FahPjNdPSe1sz9vuO3jiUyTc21DplKBv"
DIR = "/vol/lidar/office-1"
E57 = f"{DIR}/office-1.e57"
CHUNK = 2_000_000
SAMPLE = 200_000
COLOUR = ("colorRed", "colorGreen", "colorBlue")
PREFIX = 1 << 20  # voxel indices are packed 21 bits each after this offset: +-10 km at 1 cm


def fetch(path=None):
    """The E57 on the Volume, downloaded from Drive when it is not there (the ingest script's way), and committed."""
    import subprocess
    path = path or E57
    if os.path.exists(path) and os.path.getsize(path) > 1 << 20:
        with open(path, "rb") as f:
            if f.read(8) == b"ASTM-E57":
                print("E57 already on the volume", round(os.path.getsize(path) / 2**20, 1), "MB", flush=True)
                return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    subprocess.run(["curl", "-sL", "-o", path + ".part",
                    f"https://drive.usercontent.google.com/download?id={DRIVE}&export=download&confirm=t"], check=True)
    with open(path + ".part", "rb") as f:
        assert f.read(8) == b"ASTM-E57", "not an E57 (Drive sent a web page?)"
    os.replace(path + ".part", path)
    vol.commit()
    print("downloaded", round(os.path.getsize(path) / 2**20, 1), "MB in", round(time.time() - t0, 1), "s", flush=True)
    return path


def cast(n):
    """A node from .get() is generic and knows its type(); the typed classes do not, so dispatch on the class."""
    from pye57 import libe57
    T = libe57.NodeType
    kinds = {T.E57_BLOB: libe57.BlobNode, T.E57_COMPRESSED_VECTOR: libe57.CompressedVectorNode, T.E57_FLOAT: libe57.FloatNode,
             T.E57_INTEGER: libe57.IntegerNode, T.E57_SCALED_INTEGER: libe57.ScaledIntegerNode, T.E57_STRING: libe57.StringNode,
             T.E57_STRUCTURE: libe57.StructureNode, T.E57_VECTOR: libe57.VectorNode}
    for t, cls in kinds.items():
        if isinstance(n, cls):
            return t, n
    return n.type(), kinds[n.type()](n)


def describe(n):
    """A node as plain JSON: structures as dicts, vectors as lists, blobs and point tables as their sizes."""
    from pye57 import libe57
    T = libe57.NodeType
    t, n = cast(n)
    if t == T.E57_STRUCTURE:
        return {n.get(i).elementName(): describe(n.get(i)) for i in range(n.childCount())}
    if t == T.E57_VECTOR:
        return [describe(n.get(i)) for i in range(n.childCount())]
    if t == T.E57_COMPRESSED_VECTOR:
        return {"records": n.childCount()}
    if t == T.E57_BLOB:
        return {"blobBytes": n.byteCount()}
    return n.scaledValue() if t == T.E57_SCALED_INTEGER else n.value()


def pose_of(node):
    """(rotation matrix, translation) of a scan or image node, identity when it has no pose: global = R local + t."""
    import numpy as np
    from pyquaternion import Quaternion
    if not node.isDefined("pose"):
        return np.eye(3), np.zeros(3)
    p = describe(node["pose"])
    q = p.get("rotation", {"w": 1, "x": 0, "y": 0, "z": 0})
    t = p.get("translation", {"x": 0, "y": 0, "z": 0})
    return Quaternion([q["w"], q["x"], q["y"], q["z"]]).rotation_matrix, np.array([t["x"], t["y"], t["z"]], float)


def reader(e, index, fields, capacity):
    """Chunks of one scan: yields dicts of arrays (views, valid until the next chunk). Colours come as uint16 so a 16-bit
    file is not truncated; xyz as float64."""
    import numpy as np
    from pye57 import libe57
    header = e.get_header(index)
    have = set(header.point_fields)
    data, buffers = {}, libe57.VectorSourceDestBuffer()
    for f in fields:
        if f not in have:
            continue
        kind = {"colorRed": np.uint16, "colorGreen": np.uint16, "colorBlue": np.uint16, "rowIndex": np.int32, "columnIndex": np.int32,
                "intensity": np.float32, "cartesianInvalidState": np.int8, "sphericalInvalidState": np.int8}.get(f, np.float64)
        data[f] = np.empty(capacity, kind)
        buffers.append(libe57.SourceDestBuffer(e.image_file, f, data[f], capacity, True, True))
    rd = header.points.reader(buffers)
    try:
        while True:
            n = rd.read()
            if not n:
                break
            yield {k: v[:n] for k, v in data.items()}
    finally:
        rd.close()


def xyz_of(chunk, R, t):
    """Global xyz of the valid points of a chunk, and their validity mask over the chunk."""
    import numpy as np
    from pye57.utils import convert_spherical_to_cartesian
    if "cartesianX" in chunk:
        p = np.stack([chunk["cartesianX"], chunk["cartesianY"], chunk["cartesianZ"]], 1)
        bad = chunk.get("cartesianInvalidState")
    else:
        p = convert_spherical_to_cartesian(np.stack([chunk["sphericalRange"], chunk["sphericalAzimuth"], chunk["sphericalElevation"]], 1))
        bad = chunk.get("sphericalInvalidState")
    ok = np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)  # a no-return is written as (0, 0, 0) by some scanners
    if bad is not None:
        ok &= bad == 0
    return p @ R.T + t, ok


def scan_fields(header):
    cart = all(f in header.point_fields for f in ("cartesianX", "cartesianY", "cartesianZ"))
    return ["cartesianX", "cartesianY", "cartesianZ", "cartesianInvalidState"] if cart else \
           ["sphericalRange", "sphericalAzimuth", "sphericalElevation", "sphericalInvalidState"]


def image_summary(node, scans):
    """One image2D as the 8b decision needs it: the kind of camera, its size, its pose, and which scan it belongs to."""
    d = describe(node)
    kinds = [k for k in ("pinholeRepresentation", "sphericalRepresentation", "cylindricalRepresentation", "visualReferenceRepresentation") if k in d]
    rep = d[kinds[0]] if kinds else {}
    out = {"name": d.get("name"), "type": kinds[0].replace("Representation", "") if kinds else "unknown",
           "width": rep.get("imageWidth"), "height": rep.get("imageHeight"),
           "format": next((k for k in ("jpegImage", "pngImage", "imageMask") if k in rep), None),
           "bytes": next((v["blobBytes"] for k, v in rep.items() if k in ("jpegImage", "pngImage")), None),
           "pose": d.get("pose"), "scanGuid": d.get("associatedData3DGuid")}
    for k in ("focalLength", "pixelWidth", "pixelHeight", "principalPointX", "principalPointY", "radius"):
        if k in rep:
            out[k] = rep[k]
    if out["scanGuid"] in scans:
        out["scan"] = scans.index(out["scanGuid"])
    return out


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={"/vol": vol})
def inspect():
    import numpy as np
    import pye57
    path = fetch()
    e = pye57.E57(path)
    root = describe(e.root)
    scans, guids = [], []
    for i in range(e.scan_count):
        h = e.get_header(i)
        guids.append(describe(h.node["guid"]) if h.node.isDefined("guid") else None)
    for i in range(e.scan_count):
        h = e.get_header(i)
        R, t = pose_of(h.node)
        meta = {k: v for k, v in describe(h.node).items() if k not in ("points", "pose")}
        s = {"index": i, "name": meta.get("name"), "points": h.point_count, "fields": h.point_fields,
             "hasColour": all(c in h.point_fields for c in COLOUR), "hasIntensity": "intensity" in h.point_fields,
             "hasNormals": any(f.lower().startswith(("normal", "nx")) for f in h.point_fields),
             "rowColumn": "rowIndex" in h.point_fields and "columnIndex" in h.point_fields,
             "coordinates": "cartesian" if "cartesianX" in h.point_fields else "spherical",
             "stationGlobal": [round(float(v), 4) for v in t], "rotationMatrix": np.round(R, 5).tolist(),
             "localZInGlobal": np.round(R[:, 2], 4).tolist(),
             "header": {k: meta[k] for k in meta if k not in ("guid", "description")}}
        n = min(SAMPLE, h.point_count)
        take = next(reader(e, i, [*scan_fields(h), *COLOUR, "intensity", "rowIndex", "columnIndex"], n), None)
        if take is not None:
            p, ok = xyz_of(take, R, t)
            r = np.linalg.norm(p[ok] - t, axis=1)
            s["sample"] = {"points": int(len(ok)), "validFraction": round(float(ok.mean()), 4),
                           "rangeM": {q: round(float(np.percentile(r, q)), 2) for q in (1, 50, 99)} if r.size else None,
                           "boundsGlobal": [np.round(p[ok].min(0), 3).tolist(), np.round(p[ok].max(0), 3).tolist()] if ok.any() else None}
            for c in (*COLOUR, "intensity", "rowIndex", "columnIndex"):
                if c in take:
                    s["sample"][c] = [float(take[c][ok].min()), float(take[c][ok].max())] if ok.any() else None
        scans.append(s)
    imgs = root.get("images2D") or []
    n_img = len(imgs)
    summaries = [image_summary(e.root["images2D"][i], guids) for i in range(n_img)]
    kinds = {}
    for im in summaries:
        key = f"{im['type']} {im['width']}x{im['height']} {im['format']}"
        kinds[key] = kinds.get(key, 0) + 1
    report = {"file": E57, "megabytes": round(os.path.getsize(E57) / 2**20, 1), "formatName": root.get("formatName"),
              "version": [root.get("versionMajor"), root.get("versionMinor")], "coordinateMetadata": root.get("coordinateMetadata"),
              "scanCount": e.scan_count, "totalPoints": int(sum(s["points"] for s in scans)), "scans": scans,
              "images2D": {"count": n_img, "kinds": kinds, "list": summaries[:60]}}
    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/inspect.json", "w") as f:
        json.dump(report, f, indent=1)
    vol.commit()
    print("INSPECT_E57 " + json.dumps(report), flush=True)
    return report


def merge(acc, keys, pts, rgb):
    """Fold one chunk's voxels into the accumulator. acc = (keys, count, sum xyz, sum rgb), keys sorted and unique."""
    import numpy as np
    uk, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uk))
    s = np.stack([np.bincount(inv, pts[:, j], minlength=len(uk)) for j in range(3)], 1)
    c = np.stack([np.bincount(inv, rgb[:, j].astype(np.float64), minlength=len(uk)) for j in range(3)], 1)
    if acc is None:
        return uk, cnt, s, c
    k = np.concatenate([acc[0], uk])
    order = np.argsort(k, kind="stable")
    k = k[order]
    first = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])

    def fold(a, b):
        return np.add.reduceat(np.concatenate([a, b])[order], first, axis=0)

    return k[first], fold(acc[1], cnt), fold(acc[2], s), fold(acc[3], c)


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def export_points(voxel: float = 0.01):
    """Every scan in one frame (the file's global frame, minus its own centre), thinned to one point per `voxel`, as
    lidar/office-1/points-<n>cm.npz: points float32 [N,3] and colours uint8 [N,3], metres, relative to `origin` (float64);
    stations float32 [S,3] (scanner positions, same frame), station_scan int32 [S]."""
    import numpy as np
    import pye57
    path = fetch()
    e = pye57.E57(path)
    t0 = time.time()
    acc, grey, seen = None, False, 0
    stations, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
    for i in range(e.scan_count):
        h = e.get_header(i)
        R, t = pose_of(h.node)
        stations.append(t)
        colour = all(c in h.point_fields for c in COLOUR)
        grey = grey or not colour
        lim = {c: describe(h.node["colorLimits"])[c + "Maximum"] for c in COLOUR} if colour and h.node.isDefined("colorLimits") else {}
        imax = float(describe(h.node["intensityLimits"]).get("intensityMaximum", 1.0)) if h.node.isDefined("intensityLimits") else 1.0
        done = 0
        for chunk in reader(e, i, [*scan_fields(h), *COLOUR, "intensity"], CHUNK):
            p, ok = xyz_of(chunk, R, t)
            if colour:
                rgb = np.stack([chunk[c] * (255.0 / max(1.0, float(lim.get(c, 255)))) for c in COLOUR], 1)
            elif "intensity" in chunk:
                rgb = np.repeat((chunk["intensity"] / max(imax, 1e-9) * 255.0)[:, None], 3, 1)
            else:
                rgb = np.full((len(ok), 3), 128.0)
            p, rgb = p[ok], np.clip(rgb[ok], 0, 255)
            lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
            if acc is None:
                origin = np.round(p.mean(0), 3)
            q = np.floor((p - origin) / voxel).astype(np.int64)
            key = (q[:, 0] + PREFIX) | ((q[:, 1] + PREFIX) << 21) | ((q[:, 2] + PREFIX) << 42)
            acc = merge(acc, key, p - origin, rgb)
            done += len(ok)
            seen += int(ok.sum())
            print(f"scan {i}: {done}/{h.point_count} read, {len(acc[0])} voxels, {time.time() - t0:.0f} s", flush=True)
    keys, cnt, s, c = acc
    pts = (s / cnt[:, None]).astype(np.float32)
    col = np.clip(np.round(c / cnt[:, None]), 0, 255).astype(np.uint8)
    out = f"{DIR}/points-{round(voxel * 100)}cm.npz"
    np.savez(out + ".tmp.npz", points=pts, colors=col, stations=(np.array(stations) - origin).astype(np.float32),
             station_scan=np.arange(len(stations), dtype=np.int32), origin=origin, voxel=np.float32(voxel))
    os.replace(out + ".tmp.npz", out)
    vol.commit()
    report = {"file": out, "megabytes": round(os.path.getsize(out) / 2**20, 1), "pointsRead": seen, "voxels": int(len(pts)),
              "scans": e.scan_count, "origin": origin.tolist(), "boundsGlobal": [lo.round(3).tolist(), hi.round(3).tolist()],
              "colourFromIntensity": grey, "seconds": round(time.time() - t0, 1)}
    print("EXPORT_E57 " + json.dumps(report), flush=True)
    return report


@app.local_entrypoint()
def main(step: str = "inspect", voxel: float = 0.01):
    call = (inspect if step == "inspect" else export_points).spawn(*([] if step == "inspect" else [voxel]))
    print(f"[lidar] {step} started, call {call.object_id}; read it with `modal app logs ply-lidar-inspect` "
          f"(EXPORT_E57 / INSPECT_E57 line) or `modal volume ls ply-bench lidar/office-1`", flush=True)
