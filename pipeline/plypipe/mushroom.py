# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""MuSHRoom (Ren et al. 2023, Zenodo community `mushroom`, CC-BY-4.0): iPhone rooms with a Faro laser scan as reference.

What the dataset's iPhone folder holds (read from the real honka and koivu archives, 2026-11-13):
    long_capture/images/frame_NNNNN.jpg    738 x 994 portrait frames, numbered from 1 with gaps
    long_capture/depth/frame_NNNNN.png     uint16 millimetres, the same size as the frame
    long_capture/transformations.json      Polycam/ARKit poses: {"fl_x", "fl_y", "cx", "cy", "w", "h", "camera_model": "OPENCV",
                                           "frames": [{"file_path": "./images/frame_00001.jpg", "transform_matrix": 4x4, ...}]}
    long_capture/transformations_colmap.json   the same frames re-solved by COLMAP and put in metres (fewer frames; shared intrinsics)
    <room>/icp_iphone.json                 {"gt_transformation": 16 numbers}, in the reference archive (gt_mesh.ply, gt_pd.ply)
A pose is camera-to-world in the OpenGL convention (x right, y up, z backward), metres: the depth of one frame, lifted
with those poses, lands within 5 cm of the depth of a frame 4 steps on for 78% of pixels on honka, and does not under the
OpenCV convention (median error 2.9 m). The pose frame is not the scan's: `gt_transformation` joins them, and which way
it points is not documented (DN-Splatter applies its inverse to the scan to get into the pose frame, so it should take
pose-frame points to the scan frame). `orient_alignment` therefore tries both and keeps the one that puts the cameras
inside the scan, and says how sure it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plypipe import geometry

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # camera axes: OpenGL (y up, z back) -> COLMAP/OpenCV (y down, z forward)

# The two rooms (Zenodo API records read 2026-11-13, licence CC-BY-4.0 on all). The four smallest iPhone downloads are koivu (146.6 MB,
# 294 posed frames: under 300), honka (170.7 MB, 334), kokko (181.4 MB, 348) and coffee_room (183.1 MB, 388). kokko is passed over: its
# 133 ms exposure blurs the frames (median Laplacian variance 94 against honka's 574 and coffee_room's 397 on 8 frames each).
# honka is the plain room (6.1 x 3.9 x 2.3 m); coffee_room (6.3 x 5.0 x 3.1 m) shows a door and a folding partition in the one frame
# looked at, so it is taken as the room with a way into a second space (not checked against the laser scan).
ZENODO = "https://zenodo.org/api/records/{record}/files/{name}/content"
ROOMS: dict[str, dict] = {
    "honka": {"iphone": {"record": 10230733, "name": "honka_iphone.tar.gz", "bytes": 170_700_232, "md5": "f1ef5f46cd19ad0219f68d74758acf17"},
              "scan": {"record": 10222321, "name": "honka_mesh_pd.tar.gz", "bytes": 239_649_902, "md5": "75091793bf13919cb61e346300e3f828"},
              "posedFrames": 334, "sizeM": (6.1, 3.9, 2.3)},
    "coffee_room": {"iphone": {"record": 10230733, "name": "coffee_room_iphone.tar.gz", "bytes": 183_080_957, "md5": "cb467b3ad18d8f2c83dbf97a486d378b"},
                    "scan": {"record": 10222321, "name": "coffee_room_mesh_pd.tar.gz", "bytes": 288_148_502, "md5": "d051bb90179d53374c1bb0c6ec4b0745"},
                    "posedFrames": 388, "sizeM": (6.3, 5.0, 3.1)},
}


def archive_url(entry: dict) -> str:
    return ZENODO.format(record=entry["record"], name=entry["name"])


def member_target(room: str, member: str) -> tuple[str, str] | None:
    """Where a member of a room's archives goes: ("in", name) for a long-capture photo, ("raw", name) for the pose files, the test split,
    and the scan's cloud, mesh and alignment; None for everything else (depth, the short capture, Polycam's own mesh and cloud)."""
    parts = member.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "room_datasets" or parts[1] != room:
        return None
    rest, name = parts[2:], parts[-1]
    if rest[:3] == ["iphone", "long_capture", "images"] and len(rest) == 4 and name.lower().endswith(".jpg"):
        return "in", name
    if rest[:2] == ["iphone", "long_capture"] and len(rest) == 3 and name in ("transformations.json", "transformations_colmap.json", "test.txt"):
        return "raw", name
    if len(rest) == 1 and name in ("gt_pd.ply", "gt_mesh.ply", "icp_iphone.json"):
        return "raw", name
    return None


# ------------------------------------------------------------------ the dataset's files

@dataclass
class Poses:
    names: list[str]       # frame names without extension, in the file's order
    c2w: np.ndarray        # (n, 4, 4) camera-to-world, OpenCV camera axes, metres
    intrinsics: dict       # fx, fy, cx, cy, w, h of the first frame (the rooms use one camera)


def read_poses(source: str | Path | dict, opengl: bool = True) -> Poses:
    """transformations(_colmap).json -> camera-to-world matrices with OpenCV camera axes. Intrinsics may sit at the top or in each frame."""
    meta = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    frames = meta["frames"]
    if not frames:
        raise ValueError("no frames in the pose file")
    names = [Path(f["file_path"]).stem for f in frames]
    c2w = np.array([f["transform_matrix"] for f in frames], float).reshape(-1, 4, 4)
    if opengl:
        c2w = c2w @ GL_TO_CV
    first = frames[0]
    k = {key: float(first.get(src, meta.get(src))) for key, src in (("fx", "fl_x"), ("fy", "fl_y"), ("cx", "cx"), ("cy", "cy"), ("w", "w"), ("h", "h"))
         if first.get(src, meta.get(src)) is not None}
    return Poses(names, c2w, k)


def read_alignment(source: str | Path | dict) -> np.ndarray:
    """icp_iphone.json -> the 4x4 `gt_transformation` (the file stores 16 numbers or a 4x4)."""
    meta = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    return np.array(meta["gt_transformation"], float).reshape(4, 4)


def input_order(names: list[str]) -> list[str]:
    """The order `plypipe run` reads a folder of photos in (name order), whose frames it renumbers 000000.jpg, 000001.jpg ..."""
    return sorted(names)


# ------------------------------------------------------------------ alignment to the laser scan

def _nn_median(points: np.ndarray, ref: np.ndarray) -> float:
    return float(np.median(geometry.nn_distance(points, ref)))


def orient_alignment(T: np.ndarray, centres: np.ndarray, ref_points: np.ndarray) -> tuple[np.ndarray, dict]:
    """The matrix that takes pose-frame points into the scan frame: `T` or its inverse, whichever puts the camera centres nearest the scan.
    Cameras walk inside the room, so the right one lies within about a metre of a scan point; the wrong one lands metres away or outside."""
    forward, backward = geometry.apply(T, centres), geometry.apply(np.linalg.inv(T), centres)
    a, b = _nn_median(forward, ref_points), _nn_median(backward, ref_points)
    use = T if a <= b else np.linalg.inv(T)
    return use, {"direction": "as-stored" if a <= b else "inverted", "medianToScanAsStoredM": round(a, 3), "medianToScanInvertedM": round(b, 3),
                 "confident": bool(min(a, b) < 1.0 and max(a, b) > 2.0 * min(a, b))}


def scan_frame_poses(poses: Poses, to_scan: np.ndarray) -> Poses:
    """The same cameras with their matrices in the scan frame. The dataset's alignment is a similarity, not rigid: it scales the phone's
    metres to the laser's by 1.0036 on honka and 1.0096 on coffee_room (read from the archives, 2026-11-13). The centres take that scale,
    since the laser is the reference a twin's scale is scored against; the rotations take the rotation only. Past 5% it is not a
    correction of metres any more and is refused."""
    s = alignment_scale(to_scan)
    if abs(s - 1.0) > 0.05:
        raise ValueError(f"the alignment scales by {s:.4f}: the poses are metric, expected a rigid transform or one within 5% of it")
    c2w = poses.c2w.copy()
    c2w[:, :3, :3] = (to_scan[:3, :3] / s)[None] @ poses.c2w[:, :3, :3]
    c2w[:, :3, 3] = geometry.apply(to_scan, poses.c2w[:, :3, 3])
    return Poses(poses.names, c2w, poses.intrinsics)


def alignment_scale(to_scan: np.ndarray) -> float:
    """How much a pose-to-scan matrix stretches lengths (1 for a rigid one)."""
    return float(np.cbrt(abs(np.linalg.det(to_scan[:3, :3]))))


# ------------------------------------------------------------------ the twin against the reference cameras

@dataclass
class Similarity:
    scale: float           # twin units -> reference metres
    R: np.ndarray
    t: np.ndarray
    rms: float             # metres, over the inliers
    median: float          # metres, over all matched cameras
    inliers: np.ndarray    # bool per matched camera
    n: int

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.scale * self.R
        T[:3, 3] = self.t
        return T

    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)


def _residuals(src: np.ndarray, dst: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.linalg.norm(s * src @ R.T + t - dst, axis=1)


def fit_similarity(src: np.ndarray, dst: np.ndarray, trials: int = 400, keep: float = 0.8, seed: int = 0) -> Similarity:
    """Umeyama fit of twin camera centres `src` onto reference centres `dst`, robust to poses that one side got wrong (a lost track, a
    frame the laser scan's alignment does not cover). Least median of squares over random triples picks the starting fit, then the
    fit is repeated on the best `keep` share of residuals until the set stops changing (trimmed refit)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 4 or len(dst) != n:
        raise ValueError(f"need the same number (at least 4) of twin and reference centres, got {n} and {len(dst)}")
    rng = np.random.default_rng(seed)
    best, best_med = None, np.inf
    for _ in range(trials):
        i = rng.choice(n, 3, replace=False)
        s, R, t = geometry.umeyama(src[i], dst[i])
        if not np.isfinite(s) or s <= 0:
            continue
        med = float(np.median(_residuals(src, dst, s, R, t)))
        if med < best_med:
            best, best_med = (s, R, t), med
    if best is None:
        raise ValueError("no usable similarity: the camera centres are collinear or coincident")
    m = max(4, int(np.ceil(keep * n)))
    s, R, t = best
    chosen = np.zeros(n, bool)
    for _ in range(20):
        res = _residuals(src, dst, s, R, t)
        now = np.zeros(n, bool)
        now[np.argsort(res)[:m]] = True
        if (now == chosen).all():
            break
        chosen = now
        s, R, t = geometry.umeyama(src[chosen], dst[chosen])
    res = _residuals(src, dst, s, R, t)
    return Similarity(float(s), R, t, float(np.sqrt(np.mean(res[chosen] ** 2))), float(np.median(res)), chosen, n)


def scale_error_pct(sim: Similarity) -> float:
    """Signed, in percent: how much longer the twin's metres are than the room's. +8 means a wall the laser measures at 5.00 m is 5.40 m
    in the twin. Zero when the twin's scale fit onto the reference cameras is 1."""
    return float((1.0 / sim.scale - 1.0) * 100.0)


def match(twin_names: list[str], reference_names: list[str], input_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers of the twin's cameras and of the reference's that are the same frame. The pipeline renumbers a folder of photos
    000000.jpg, 000001.jpg ... in name order (`input_order`), so a twin name's number indexes the sorted names of the folder it was
    given (`input_names`, default the reference's own: the folder may hold frames the reference has no pose for); a name in the
    reference's own style (frame_00012) matches by name."""
    ref = {n: i for i, n in enumerate(reference_names)}
    order = input_order(input_names if input_names is not None else reference_names)
    rows_t, rows_r = [], []
    for k, name in enumerate(twin_names):
        stem = Path(name).stem
        if stem in ref:
            rows_t.append(k), rows_r.append(ref[stem])
        elif stem.isdigit() and int(stem) < len(order) and order[int(stem)] in ref:
            rows_t.append(k), rows_r.append(ref[order[int(stem)]])
    return np.array(rows_t, int), np.array(rows_r, int)


def match_and_fit(twin_names: list[str], twin_centres: np.ndarray, reference_names: list[str], reference_centres: np.ndarray,
                  input_names: list[str] | None = None) -> Similarity:
    rt, rr = match(twin_names, reference_names, input_names)
    if len(rt) < 4:
        raise ValueError(f"only {len(rt)} of the twin's {len(twin_names)} cameras match a reference frame")
    return fit_similarity(np.asarray(twin_centres)[rt], np.asarray(reference_centres)[rr])


def reference_to_twin(sim: Similarity, points: np.ndarray) -> np.ndarray:
    """Reference points (a mesh's vertices, the laser cloud) in the twin's frame."""
    return geometry.apply(sim.inverse(), points)


def twin_to_reference(sim: Similarity, points: np.ndarray) -> np.ndarray:
    return geometry.apply(sim.matrix, points)


# ------------------------------------------------------------------ files on the Volume

_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
              "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def read_ply_points(path: str | Path, stride: int = 1, limit: int | None = None) -> np.ndarray:
    """x, y, z of every `stride`-th vertex of a binary PLY (a mesh's too: the vertex block comes first), float32, read through a memory map
    so a 650 MB laser mesh costs its slice and not its size. `limit`: widen the stride until at most that many points come back."""
    path = Path(path)
    with path.open("rb") as f:
        lines, size = [], 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            size += len(line)
            lines.append(line.decode("latin-1").split())
            if lines[-1][:1] == ["end_header"]:
                break
    fmt = next((ln[1] for ln in lines if ln[:1] == ["format"]), "")
    if not fmt.startswith("binary"):
        raise ValueError(f"{path}: {fmt or 'unknown'} PLY; only binary is supported")
    end = "<" if fmt == "binary_little_endian" else ">"
    count, fields, seen = 0, [], False
    for ln in lines:
        if ln[:2] == ["element", "vertex"]:
            count, seen = int(ln[2]), True
        elif ln[:1] == ["element"]:
            seen = False
        elif ln[:1] == ["property"] and seen and ln[1] != "list":
            fields.append((ln[2], end + _PLY_TYPES[ln[1]]))
    if not {"x", "y", "z"} <= {n for n, _ in fields}:
        raise ValueError(f"{path}: the vertex element has no x, y, z")
    data = np.memmap(path, dtype=np.dtype(fields), mode="r", offset=size, shape=(count,))
    sel = data[:: max(1, int(stride), -(-count // limit) if limit else 1)]
    return np.stack([np.asarray(sel[a], np.float32) for a in "xyz"], axis=1)


def thin(points: np.ndarray, cell: float, limit: int | None = None, seed: int = 0) -> np.ndarray:
    """One point per `cell`-metre voxel (the first seen), then at most `limit` of them chosen at random."""
    ijk = np.floor(points / cell).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    _, first = np.unique((ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2], return_index=True)  # one int64 per voxel: fast for tens of millions
    out = points[np.sort(first)]
    if limit and len(out) > limit:
        out = out[np.random.default_rng(seed).choice(len(out), limit, replace=False)]
    return out


def read_twin_cameras(sparse_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Names and world centres of the cameras in a COLMAP text model (the job's data/sparse/0, which is in the scene frame)."""
    from plypipe import colmap_txt
    model = colmap_txt.read_model(sparse_dir)
    ims = sorted(model.images, key=lambda im: im.name)
    centres, _, _ = model.cameras()
    return [im.name for im in ims], centres


def build_reference(pose_files: dict[str, Path], alignment: Path, scan: np.ndarray, input_names: list[str], cell: float = 0.02,
                    limit: int = 1_500_000) -> tuple[dict, np.ndarray]:
    """What a run is scored against, from the dataset's files: the laser cloud thinned to one point per `cell` metres, and for the pose
    file whose cameras sit nearest the scan (honka and kokko have two: ARKit's and COLMAP's, in one frame) their names and centres in
    the scan frame, with how the alignment was oriented. Returns (meta for meta.json, the thinned cloud)."""
    cloud = thin(np.asarray(scan, np.float32), cell, limit)
    T = read_alignment(alignment)
    tried = []
    for label, path in pose_files.items():
        poses = read_poses(path)
        to_scan, info = orient_alignment(T, poses.c2w[:, :3, 3], cloud)
        aligned = scan_frame_poses(poses, to_scan)
        centres = aligned.c2w[:, :3, 3]
        tried.append({"file": label, "frames": len(poses.names), "medianToScanM": round(_nn_median(centres, cloud), 3), **info,
                      "alignmentScale": round(alignment_scale(to_scan), 5),
                      "names": poses.names, "centres": centres})
    pick = min(tried, key=lambda r: r["medianToScanM"])
    meta = {"poses": pick["file"], "names": pick["names"], "centres": np.round(pick["centres"], 5).tolist(), "inputNames": sorted(input_names),
            "direction": {k: pick[k] for k in ("direction", "medianToScanAsStoredM", "medianToScanInvertedM", "confident")},
            "candidates": [{k: v for k, v in r.items() if k not in ("names", "centres")} for r in tried],
            "scanPoints": int(len(cloud)), "cellM": cell, "alignment": T.tolist()}
    return meta, cloud


def load_reference(folder: str | Path) -> dict:
    """What mushroom_seed.py leaves in <room>/reference: meta.json (frame names and scan-frame centres, the input folder's names, how the
    alignment was oriented) and ref_points.npy, the thinned scan cloud."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    return {"meta": meta, "names": meta["names"], "inputNames": meta.get("inputNames"), "centres": np.array(meta["centres"], float),
            "points": np.load(folder / "ref_points.npy")}


def score_job(job: str | Path, reference: dict, splat_alpha: float = 0.5, max_twin: int = 400_000, splats: bool = True) -> dict:
    """The laser-reference scores of a finished job: the twin's metric scale (from its camera solve) and, when the job has a splat and
    `splats` is true, how well the splat's centres lie on the scan (precision and recall at 5 and 10 cm, chamfer), in the twin's own frame.
    The nearest-point search runs on the GPU when torch finds one (400k splats against a 1.5M-point scan is minutes on 4 CPU cores)."""
    job = Path(job)
    names, centres = read_twin_cameras(job / "data" / "sparse" / "0")
    sim = match_and_fit(names, centres, reference["names"], reference["centres"], reference.get("inputNames"))
    out = {"scaleErrorPct": round(scale_error_pct(sim), 2), "scale": round(sim.scale, 4), "cameras": {"twin": len(names), "matched": sim.n,
           "inliers": int(sim.inliers.sum()), "rmsM": round(sim.rms, 3), "medianM": round(sim.median, 3)},
           "reference": {"poses": reference["meta"].get("poses"), "alignment": reference["meta"].get("direction")}}
    ply = job / "scene.ply"
    if splats and ply.exists():
        from plypipe import topview
        s = topview.read_ply(ply)
        keep = s.alpha >= splat_alpha
        pts = np.stack([s.x[keep], s.y[keep], s.z[keep]], axis=1).astype(np.float64)
        if len(pts) > max_twin:
            pts = pts[np.random.default_rng(0).choice(len(pts), max_twin, replace=False)]
        out["splats"] = {"points": int(len(pts)), **geometry.cloud_scores(pts, reference_to_twin(sim, reference["points"].astype(np.float64)))}
    return out
