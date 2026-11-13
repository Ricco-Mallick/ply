# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""sfm: frames -> camera poses + sparse points (COLMAP 4.2.1 via pycolmap), undistorted to PINHOLE.

Features: ALIKED (N16ROT) + LightGlue by default, falling back to SIFT if the ONNX models cannot
be loaded. One SIMPLE_RADIAL camera shared by every frame (a phone video is one camera).
Pairs: exhaustive for small captures (<= exhaustive_max frames, best loop closure), sequential
with quadratic overlap otherwise (video order is a strong prior).
Mapper: global SfM (GLOMAP, merged into COLMAP 4.x), keeping the largest model; ordered video of
incremental_min frames or more goes straight to the incremental mapper (see choose_mapper). Two faster solves
for long captures are opt-in (--mapper): hierarchical (clusters mapped in parallel, merged, one bundle adjustment)
and keyframes (every few frames solved first, the rest registered against them with their poses fixed).

Outputs:
    colmap/database.db, colmap/sparse/0          raw model (SIMPLE_RADIAL, binary)
    dense/images, dense/sparse                    undistorted PINHOLE images + binary model
    dense/sparse_txt                              same model as text, read by `align`
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

from plypipe.util import Ctx

# COLMAP's own retrieval trees (the URIs pycolmap 4.2.1 carries), one per descriptor. They are fetched here rather
# than by COLMAP: its downloader aborts the whole process on a TLS error instead of raising.
VOCAB_TREES = {
    "aliked": ("https://github.com/colmap/colmap/releases/download/3.13.0/vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin",
               "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e"),
    "sift": ("https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin",
             "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"),
}
ORDERED_SOURCES = ("video", "videos", "ar")
FAST_MAPPERS = ("hierarchical", "keyframes")  # opt-in; checked and re-solved incrementally like the global mapper


def vocab_tree(features: str, cache: Path | None = None) -> Path | None:
    """Local copy of the retrieval tree for these features (sha256-checked), or None if it cannot be fetched."""
    url, sha = VOCAB_TREES[features]
    cache = Path(cache or os.environ.get("PLY_CACHE_DIR") or Path.home() / ".cache" / "plypipe")
    dest = cache / url.rsplit("/", 1)[1]
    if dest.exists():
        return dest
    try:
        cache.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(url, timeout=300).read()
        if hashlib.sha256(data).hexdigest() != sha:
            return None
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return dest
    except OSError:
        return None


def _pycolmap():
    try:
        import pycolmap  # noqa: F401
    except ImportError as e:  # pragma: no cover - needs the GPU env
        raise RuntimeError(f"pycolmap could not be imported ({type(e).__name__}: {e}). On a CUDA 12 machine: "
                           "pip install 'pycolmap-cuda12==4.2.1' (CPU-only: pip install 'pycolmap==4.2.1').") from e
    return pycolmap


def focal_prior_params(images: Path, focal_35mm: float) -> str:
    """SIMPLE_RADIAL params "f,cx,cy,k" from a 35 mm-equivalent focal length.

    Video frames carry no EXIF, so COLMAP would start from f = 1.2 * max(w, h), and a capture
    with little rotation variety can converge to a wrong focal length, which tilts the whole
    reconstruction. Phone specs quote the equivalent focal on the diagonal (43.27 mm).
    """
    import cv2

    first = next(iter(sorted(images.glob("*.jpg"))))
    h, w = cv2.imread(str(first)).shape[:2]
    f = focal_35mm / 43.27 * (w * w + h * h) ** 0.5
    return f"{f:.3f},{w / 2:.3f},{h / 2:.3f},0"


_PROBE = r"""
import sys, pycolmap
ext = pycolmap.FeatureExtractionOptions(pycolmap.FeatureExtractorType.ALIKED_N16ROT)
ext.max_image_size = 320
pycolmap.extract_features(sys.argv[1], sys.argv[2], extraction_options=ext, device=pycolmap.Device(sys.argv[3]))
"""


def aliked_available(images: Path, device: str = "auto") -> bool:
    """ALIKED needs a pycolmap built with ONNX. Without it COLMAP aborts the process instead of raising (the
    pycolmap-cuda12 4.2.1 wheel does exactly that), so one frame is tried in a throwaway process first."""
    import subprocess
    import sys
    import tempfile
    first = next(iter(sorted(images.glob("*.jpg"))), None)
    if first is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "img").mkdir()
        shutil.copy2(first, Path(tmp) / "img" / first.name)
        try:
            r = subprocess.run([sys.executable, "-c", _PROBE, str(Path(tmp) / "probe.db"), str(Path(tmp) / "img"), device],
                               capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def _camera_mode(pycolmap, images: Path):
    """One shared camera when every frame has the same size (one phone or one action cam); per-image cameras when
    clips were shot at different resolutions or aspect ratios (a single camera would be wrong for some of them)."""
    import cv2
    sizes = {cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape for p in sorted(images.glob("*.jpg"))}
    return pycolmap.CameraMode.SINGLE if len(sizes) == 1 else pycolmap.CameraMode.PER_IMAGE


def _extract(pycolmap, db: Path, images: Path, features: str, device, camera_params: str | None = None,
             camera_model: str = "SIMPLE_RADIAL") -> str:
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    if camera_params and camera_model == "SIMPLE_RADIAL":  # the focal prior is written for SIMPLE_RADIAL's parameters
        reader.camera_params = camera_params
    if features == "aliked":
        ext = pycolmap.FeatureExtractionOptions(pycolmap.FeatureExtractorType.ALIKED_N16ROT)
    else:
        ext = pycolmap.FeatureExtractionOptions(pycolmap.FeatureExtractorType.SIFT)
    ext.max_image_size = 1600
    pycolmap.extract_features(db, images, camera_mode=_camera_mode(pycolmap, images), reader_options=reader,
                              extraction_options=ext, device=device)
    return features


def _match(pycolmap, db: Path, features: str, n_images: int, matcher: str, exhaustive_max: int, device,
           tree: Path | None = None, unordered: bool = False) -> str:
    """Pairs and matches. Video: sequential, plus loop closure by retrieval when a tree is available (walking back
    into a room is a loop the neighbours never see). Photos: file order says nothing about overlap, so exhaustive
    while affordable, then retrieval."""
    mtype = (pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE if features == "aliked"
             else pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE)
    mopts = pycolmap.FeatureMatchingOptions(mtype)
    if matcher == "auto":
        matcher = ("exhaustive" if n_images <= exhaustive_max
                   else "vocabtree" if unordered and tree else "sequential")
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=mopts, device=device)
    elif matcher == "vocabtree":
        if tree is None:
            raise RuntimeError("vocabtree matching needs the retrieval tree, which could not be downloaded")
        pairing = pycolmap.VocabTreePairingOptions()
        pairing.vocab_tree_path = str(tree)
        pairing.num_images = 50
        pycolmap.match_vocabtree(db, matching_options=mopts, pairing_options=pairing, device=device)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 20
        pairing.quadratic_overlap = True
        if tree is not None:
            pairing.loop_detection = True
            pairing.vocab_tree_path = str(tree)
        pycolmap.match_sequential(db, matching_options=mopts, pairing_options=pairing, device=device)
    return matcher


def drop_unanchored(rec, min_points: int = 20, far: float = 20.0) -> list[str]:
    """Deregister views the solve could not pin down; returns their names.

    The global mapper keeps a view registered after its observations are filtered out, at whatever position global
    positioning left it: one 0-point view of the Dr Johnson demo sat 4 million median distances away. The model's
    reprojection error never sees it, but gsplat sizes its learning rates from the farthest camera, so a single such
    view makes training diverge. Views with fewer than `min_points` 3D points, or more than `far` times the median
    camera distance from the median centre, are dropped."""
    import numpy as np
    dropped = []
    for im in list(rec.images.values()):
        if im.has_pose and im.num_points3D < min_points:
            dropped.append(im.name)
            rec.deregister_frame(im.frame_id)
    posed = [im for im in rec.images.values() if im.has_pose]
    if len(posed) >= 5:
        centres = np.array([im.projection_center() for im in posed])
        dist = np.linalg.norm(centres - np.median(centres, axis=0), axis=1)
        med = float(np.median(dist))
        for im, d in zip(posed, dist):
            if med > 0 and d > far * med:
                dropped.append(im.name)
                rec.deregister_frame(im.frame_id)
    return dropped


def _best(models: dict):
    return max(models.items(), key=lambda kv: kv[1].num_reg_images())[1] if models else None


def clip_index(out: Path) -> dict[str, str] | None:
    """frame name -> source clip, from the frames stage (None for photo sets and older jobs)."""
    try:
        return {d["frame"]: d["clip"] for d in json.loads((out / "frames_index.json").read_text())}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _path_steps(rec, clip_of: dict[str, str] | None):
    """Consecutive posed views (name order) and the step between each pair. A step that crosses from one clip to the
    next is not a step of one walk, so it is left out: the next clip can start anywhere in the room."""
    import numpy as np
    posed = [im for im in sorted(rec.images.values(), key=lambda im: im.name) if im.has_pose]
    centres = [np.asarray(im.projection_center(), dtype=float) for im in posed]
    clip = (lambda im: clip_of.get(im.name)) if clip_of else (lambda im: None)
    steps = [(i, float(np.linalg.norm(centres[i + 1] - centres[i]))) for i in range(len(posed) - 1)
             if clip(posed[i]) == clip(posed[i + 1])]
    return posed, centres, steps, clip


def trajectory_jumps(rec, ratio: float = 8.0, clip_of: dict[str, str] | None = None) -> int:
    """Video frames are sequential, so consecutive camera centres must move smoothly. A step far
    larger than the median step means the model was stitched from inconsistent pieces, even
    when the reprojection error looks excellent."""
    import numpy as np
    _, _, steps, _ = _path_steps(rec, clip_of)
    if len(steps) < 4:
        return 0
    d = np.array([v for _, v in steps])
    med = float(np.median(d)) or 1e-12
    return int((d > ratio * med).sum())


def drop_spikes(rec, ratio: float = 8.0, clip_of: dict[str, str] | None = None) -> list[str]:
    """Deregister single views the solve placed far off the walk, out and back between two neighbours that sit
    together; returns their names.

    On the chapel, three frames looking down rows of identical pews landed 22-66 median steps from both neighbours
    while those neighbours were one step apart: a pose matched to the wrong pew, which training then bakes in as a
    ghost. Counted as path jumps they also sent a good solve to the slow fallback."""
    import numpy as np
    posed, centres, steps, clip = _path_steps(rec, clip_of)
    if len(steps) < 4:
        return []
    med = float(np.median([v for _, v in steps])) or 1e-12
    dropped = []
    for i in range(1, len(posed) - 1):
        if not (clip(posed[i - 1]) == clip(posed[i]) == clip(posed[i + 1])):
            continue
        a = float(np.linalg.norm(centres[i] - centres[i - 1]))
        b = float(np.linalg.norm(centres[i + 1] - centres[i]))
        if a > ratio * med and b > ratio * med and np.linalg.norm(centres[i + 1] - centres[i - 1]) < 0.25 * min(a, b):
            dropped.append(posed[i].name)
    for im in posed:
        if im.name in dropped:
            rec.deregister_frame(im.frame_id)
    return dropped


def choose_mapper(requested: str | None, source: str | None, n_images: int, incremental_min: int = 400) -> tuple[str, str]:
    """The mapper to run first and why. An explicit choice always wins. Without one, a long ordered walk skips the
    global mapper: it was rejected on all five chapel solves (21 to 83 path jumps), 32 minutes before the fallback."""
    if requested:
        return requested, "requested"
    if source in ORDERED_SOURCES and n_images >= incremental_min:
        return "incremental", f"{source} of {n_images} frames, at least {incremental_min}"
    return "global", "default"


def incremental_options(pycolmap, n_images: int, max_runtime_s: int | None = None):
    """Incremental mapper settings. On long videos global bundle adjustment dominates (the chapel's 1,236 frames:
    90 min on 8 cores), so it runs less often and skips 3D points that add no coverage."""
    opts = pycolmap.IncrementalPipelineOptions()
    if max_runtime_s:
        opts.max_runtime_seconds = int(max_runtime_s)
    if n_images > 400:
        opts.ba_global_frames_ratio = 1.25
        opts.ba_global_points_ratio = 1.25
        opts.mapper.ba_global_ignore_redundant_points3D = True
    return opts


def select_keyframes(names: list[str], clip_of: dict[str, str] | None = None, step: int = 4, max_gap: int = 8,
                     linked=None) -> list[str]:
    """Frames to solve first: every `step`-th frame of each clip plus its first and last, so no clip boundary is lost.
    A gap wider than `max_gap`, or between two frames `linked(a, b)` says share no verified matches, is halved until it
    is not: sequential matching pairs frame i only with i+1, i+2, i+4, i+8 ... (measured on a 120-frame scene: 118 pairs
    at offset 1, 113 at 2, 11 at 3), so keyframes three apart have no direct match and the solve falls apart at 2 of 41."""
    keys: list[str] = []
    runs: list[list[str]] = []
    for name in sorted(names):
        if runs and (clip_of.get(name) if clip_of else None) == (clip_of.get(runs[-1][-1]) if clip_of else None):
            runs[-1].append(name)
        else:
            runs.append([name])
    for run in runs:
        marks = sorted(set(range(0, len(run), max(1, step))) | {len(run) - 1})
        out = [marks[0]]
        for b in marks[1:]:
            todo = [(out[-1], b)]
            while todo:
                lo, hi = todo.pop()
                if hi - lo > 1 and (hi - lo > max_gap or (linked is not None and not linked(run[lo], run[hi]))):
                    mid = (lo + hi) // 2
                    todo += [(mid, hi), (lo, mid)]
                else:
                    out.append(hi)
        keys += [run[i] for i in out]
    return keys


def linked_pairs(pycolmap, db: Path, min_inliers: int = 30):
    """linked(a, b): do these two frames share at least `min_inliers` geometrically verified matches in the database?"""
    d = pycolmap.Database.open(str(db))
    try:
        ids = {im.name: im.image_id for im in d.read_all_images()}
        pair_ids, counts = d.read_two_view_geometry_num_inliers()
    finally:
        d.close()
    good = {int(p) for p, c in zip(pair_ids, counts) if c >= min_inliers}
    return lambda a, b: (min(ids[a], ids[b]) * 2147483647 + max(ids[a], ids[b])) in good


def _final_bundle(pycolmap, rec) -> None:
    """One bundle adjustment over everything, intrinsics and points free (the incremental mapper's own setting)."""
    pycolmap.bundle_adjustment(rec, pycolmap.BundleAdjustmentOptions())


def _map_hierarchical(ctx: Ctx, pycolmap, db: Path, images: Path, out: Path, n_images: int, seconds: dict):
    """Clusters of at most --hier-leaf-max frames are mapped in parallel and merged; the merge does no bundle
    adjustment (the 120-frame test scene came out at 0.236 px, 0.186 px after one), so one runs over the biggest model."""
    leaf = int(ctx.opt("hier_leaf_max", 500))
    opts = pycolmap.HierarchicalPipelineOptions()
    opts.clustering_options.leaf_max_num_images = leaf
    opts.incremental_options = incremental_options(pycolmap, min(n_images, leaf), ctx.opt("incremental_max_s"))
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    models = pycolmap.hierarchical_mapping(db, images, out, opts)
    seconds["hierarchical"] = round(time.time() - t, 1)
    best = _best(models)
    if best is not None:
        t = time.time()
        _final_bundle(pycolmap, best)
        seconds["bundle"] = round(time.time() - t, 1)
    return models


def _map_keyframes(ctx: Ctx, pycolmap, db: Path, images: Path, out: Path, n_images: int, clips: dict[str, str] | None,
                   seconds: dict, extra: dict):
    """Keyframes first, then the other frames registered against that model with the keyframes' poses held (the
    mapper's fix_existing_frames; naming only the rest in image_names registered 119 of 120 frames, so the list is left empty), then one bundle
    adjustment over everything. On the 120-frame test scene the keyframes' poses moved by 3e-15 m in the register step."""
    names = sorted(p.name for p in images.glob("*.jpg"))
    keys = select_keyframes(names, clips, int(ctx.opt("keyframe_step", 4)), int(ctx.opt("keyframe_max_gap", 8)),
                            linked_pairs(pycolmap, db))
    extra["keyframes"] = len(keys)
    out.mkdir(parents=True, exist_ok=True)
    ctx.log.emit("sfm", "progress", f"solving {len(keys)} keyframes of {n_images} first")
    opts = incremental_options(pycolmap, len(keys), ctx.opt("incremental_max_s"))
    opts.image_names = keys
    t = time.time()
    first = _best(pycolmap.incremental_mapping(db, images, out / "keyframes", opts))
    seconds["keyframes"] = round(time.time() - t, 1)
    if first is None:
        return {}
    kdir = out / "keyframes" / "best"
    kdir.mkdir(parents=True, exist_ok=True)
    first.write(kdir)
    ctx.log.emit("sfm", "progress", f"{first.num_reg_images()} keyframes solved: registering the other frames")
    opts = incremental_options(pycolmap, n_images, ctx.opt("incremental_max_s"))
    opts.fix_existing_frames = True
    t = time.time()
    models = pycolmap.incremental_mapping(db, images, out / "registered", opts, input_path=str(kdir))
    seconds["register"] = round(time.time() - t, 1)
    best = models.get(0)  # the extended keyframe model; later ids are new models grown from frames it could not take
    if best is None:
        return {}
    t = time.time()
    _final_bundle(pycolmap, best)
    seconds["bundle"] = round(time.time() - t, 1)
    return {0: best}


def run_stage(ctx: Ctx) -> dict:
    pycolmap = _pycolmap()
    images = ctx.frames_dir
    n_images = len(list(images.glob("*.jpg")))
    if n_images < 10:
        raise RuntimeError(f"only {n_images} frames: need at least 10 (check the frames stage)")
    device = pycolmap.Device(ctx.opt("device", "auto"))
    cdir = ctx.colmap_dir
    db = cdir / "database.db"
    source = None
    try:
        source = json.loads((ctx.out / ".done" / "frames.json").read_text()).get("source")
    except (OSError, ValueError):
        pass
    requested = ctx.opt("features", "aliked")
    cam_params = focal_prior_params(images, float(ctx.opt("focal_35mm"))) if ctx.opt("focal_35mm") else None
    if ctx.opt("reuse_matches") and db.exists():
        # re-solve from the matches already in the database (a crashed or re-tuned mapping): matching the chapel's
        # 1,236 frames took 10 min of GPU time, mapping is the part being redone
        for sub in ("sparse_all", "sparse_inc", "sparse"):
            shutil.rmtree(cdir / sub, ignore_errors=True)
        prev = {}
        for f in (cdir / "matching.json", ctx.out / ".done" / "sfm.json"):
            try:
                prev = json.loads(f.read_text())
                break
            except (OSError, ValueError):
                continue
        features, matcher, loops = prev.get("features", "reused"), prev.get("matcher", "reused"), prev.get("loopDetection")
        ctx.log.emit("sfm", "progress", f"reusing the {features} matches in {db.name}", images=n_images)
    else:
        features, matcher, loops = _extract_and_match(ctx, pycolmap, cdir, db, images, n_images, device, requested, source,
                                                      cam_params)
        if ctx.opt("sfm_part") == "match" and n_images >= int(ctx.opt("split_min", 400) or 0):
            # the GPU half of a split solve: features and matches are in the database; a CPU box maps them later
            # (`--reuse-matches`). The stage is not marked done, so the next run resumes here.
            return {"partial": "matched", "features": features, "matcher": matcher, "loopDetection": loops,
                    "images": n_images}
    return _map(ctx, pycolmap, cdir, db, images, n_images, source, features, requested, matcher, loops, cam_params)


def _extract_and_match(ctx: Ctx, pycolmap, cdir: Path, db: Path, images: Path, n_images: int, device, requested: str,
                       source: str | None, cam_params: str | None):
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    features = requested
    if features == "aliked-lg":
        # opt-in, not the default: on the chapel its 16M matches made the incremental mapper's bundle adjustment run
        # 50+ min without finishing, and the global mapper broke in 83 places against SIFT's 27 (docs/IMPROVEMENT.md)
        from plypipe.stages import sfm_torch
        if sfm_torch.available():
            ctx.log.emit("sfm", "progress", "ALIKED + LightGlue in PyTorch (this pycolmap has no ONNX runtime)", images=n_images)
            reader = pycolmap.ImageReaderOptions()
            reader.camera_model = ctx.opt("camera_model", "SIMPLE_RADIAL")
            if cam_params and reader.camera_model == "SIMPLE_RADIAL":
                reader.camera_params = cam_params
            stats = sfm_torch.extract_and_match(
                pycolmap, db, images, _camera_mode(pycolmap, images), reader,
                clip_index(ctx.out) if source in ORDERED_SOURCES else None, ordered=source in ORDERED_SOURCES,
                log=lambda msg, **kw: ctx.log.emit("sfm", "progress", msg, **kw))
            (cdir / "matching.json").write_text(json.dumps({"features": "aliked", "matcher": "lightglue", "loopDetection": stats["retrieval"], "torch": stats}))
            ctx.log.emit("sfm", "progress", "matched", **stats)
            return "aliked", "lightglue", stats["retrieval"]
        ctx.log.emit("sfm", "warn", "aliked-lg needs a CUDA GPU and the lightglue package: using SIFT")
        features = "sift"
    if features == "aliked" and not aliked_available(images, ctx.opt("device", "auto")):
        ctx.log.emit("sfm", "warn", "this pycolmap has no ONNX runtime, so ALIKED would abort the process: using SIFT")
        features = "sift"
    ctx.log.emit("sfm", "progress", f"extracting {features} features", images=n_images, cameraPrior=cam_params)
    try:
        _extract(pycolmap, db, images, features, device, cam_params, ctx.opt("camera_model", "SIMPLE_RADIAL"))
    except Exception as e:  # ONNX model download/runtime missing -> SIFT
        if features != "aliked":
            raise
        ctx.log.emit("sfm", "warn", f"ALIKED failed ({e}); falling back to SIFT")
        db.unlink(missing_ok=True)
        features = "sift"
        _extract(pycolmap, db, images, features, device, cam_params, ctx.opt("camera_model", "SIMPLE_RADIAL"))

    ctx.log.emit("sfm", "progress", "matching")
    # several clips: frames from different clips sit next to each other in the numbering, so sequential pairing
    # would try to join unrelated views and miss the real overlaps; pair exhaustively while that stays affordable
    exhaustive_max = int(ctx.opt("exhaustive_max", 250))
    if source == "videos":
        exhaustive_max = max(exhaustive_max, 700)
    elif source == "images" and exhaustive_max == 250:  # 250 = the CLI default: an explicit value is respected
        exhaustive_max = 500  # a photo set has no order to exploit; 500 photos is ~125k pairs, minutes on a GPU
    elif source == "ar" and exhaustive_max == 250:
        # AR keyframes come in walking order, many of them: sequential pairing with quadratic overlap finds the
        # loops too, at a fraction of exhaustive matching's cost (n^2 pairs took 20 min for 146 frames on a T4)
        exhaustive_max = 150
    unordered = source == "images"
    want = ctx.opt("matcher", "auto")
    needs_tree = not ctx.opt("no_loop_detection") and (want in ("sequential", "vocabtree") or
                                                       (want == "auto" and n_images > exhaustive_max))

    def tree_for(feat: str) -> Path | None:
        if not needs_tree:
            return None
        t = vocab_tree(feat)
        if t is None:
            ctx.log.emit("sfm", "warn", f"retrieval tree for {feat} could not be fetched: matching without loop closure")
        return t

    tree = tree_for(features)
    try:
        matcher = _match(pycolmap, db, features, n_images, want, exhaustive_max, device, tree, unordered)
    except Exception as e:
        if features != "aliked":
            raise
        ctx.log.emit("sfm", "warn", f"LightGlue matching failed ({e}); re-running with SIFT")
        db.unlink(missing_ok=True)
        features = "sift"
        _extract(pycolmap, db, images, features, device, cam_params, ctx.opt("camera_model", "SIMPLE_RADIAL"))
        tree = tree_for(features)
        matcher = _match(pycolmap, db, features, n_images, want, exhaustive_max, device, tree, unordered)
    loops = matcher == "sequential" and tree is not None
    (cdir / "matching.json").write_text(json.dumps({"features": features, "matcher": matcher, "loopDetection": loops}))
    return features, matcher, loops


def _map(ctx: Ctx, pycolmap, cdir: Path, db: Path, images: Path, n_images: int, source: str | None, features: str,
         requested: str, matcher: str, loops, cam_params: str | None):
    mapper, mapper_why = choose_mapper(ctx.opt("mapper"), source, n_images, int(ctx.opt("incremental_min", 400)))
    ctx.log.emit("sfm", "progress", f"{mapper} mapping ({mapper_why})")
    raw_out = cdir / "sparse_all"
    ordered = source in ORDERED_SOURCES
    clips = clip_index(ctx.out) if ordered else None
    seconds: dict[str, float] = {}
    extra: dict = {}
    t0 = time.time()
    if mapper == "incremental":
        # no time cap unless asked: there is no other model to fall back on, so a cut-off would only ship a partial one
        models = pycolmap.incremental_mapping(db, images, raw_out,
                                              incremental_options(pycolmap, n_images, ctx.opt("incremental_max_s")))
    elif mapper == "hierarchical":
        models = _map_hierarchical(ctx, pycolmap, db, images, raw_out, n_images, seconds)
    elif mapper == "keyframes":
        models = _map_keyframes(ctx, pycolmap, db, images, raw_out, n_images, clips, seconds, extra)
    else:
        models = pycolmap.global_mapping(db, images, raw_out, pycolmap.GlobalPipelineOptions())
    seconds = seconds or {"mapper": round(time.time() - t0, 1)}
    if not models and not (mapper in FAST_MAPPERS and not ctx.opt("no_mapper_fallback")):
        raise RuntimeError(f"{mapper} mapping produced no model: capture lacks overlap or texture (see docs/SOP.md)")
    best = _best(models)

    def clean(rec) -> tuple[list[str], list[str], int]:
        unanchored = drop_unanchored(rec)
        spikes = drop_spikes(rec, clip_of=clips) if ordered else []
        return unanchored, spikes, (trajectory_jumps(rec, clip_of=clips) if ordered else 0)

    dropped, spikes, jumps = clean(best) if best is not None else ([], [], 0)
    if dropped:
        ctx.log.emit("sfm", "warn", f"{len(dropped)} views had no anchor in the solve (too few points, or far from "
                     f"every other camera) and were dropped: {', '.join(dropped[:8])}{' …' if len(dropped) > 8 else ''}")
    if spikes:
        ctx.log.emit("sfm", "warn", f"{len(spikes)} views sat far off the camera path between two neighbours "
                     f"(mismatched poses) and were dropped: {', '.join(spikes[:8])}{' …' if len(spikes) > 8 else ''}")
    # the global mapper can fold a video walk into a model with an excellent reprojection error (glomap#145);
    # a camera path that jumps, or a solve that drops views, is re-solved incrementally and the better model kept
    fallback = None
    reg_ratio = best.num_reg_images() / n_images if best is not None else 0.0
    if mapper in ("global", *FAST_MAPPERS) and not ctx.opt("no_mapper_fallback") and (jumps or reg_ratio < 0.9):
        why = f"{jumps} path jumps" if jumps else f"{reg_ratio:.0%} registered"
        ctx.log.emit("sfm", "warn", f"{mapper} solve looks wrong ({why}): solving again with the incremental mapper")
        opts = incremental_options(pycolmap, n_images, int(ctx.opt("incremental_max_s", 5400)))
        t0 = time.time()
        inc = pycolmap.incremental_mapping(db, images, cdir / "sparse_inc", opts)
        seconds["fallback"] = round(time.time() - t0, 1)
        alt = _best(inc)
        fallback = {"reason": why, mapper: {"registered": best.num_reg_images() if best is not None else 0, "jumps": jumps}, "used": mapper}
        if alt is not None:
            alt_dropped, alt_spikes, alt_jumps = clean(alt)
            fallback["incremental"] = {"registered": alt.num_reg_images(), "jumps": alt_jumps}
            score = lambda m, j: (j == 0 and m.num_reg_images() >= 0.6 * n_images, m.num_reg_images())
            if best is None or score(alt, alt_jumps) > score(best, jumps):
                best, jumps, models, dropped, spikes, fallback["used"] = alt, alt_jumps, inc, alt_dropped, alt_spikes, "incremental"
        ctx.log.emit("sfm", "progress", f"kept the {fallback['used']} model")
    if best is None:
        raise RuntimeError(f"{mapper} mapping produced no model: capture lacks overlap or texture (see docs/SOP.md)")
    if jumps:
        ctx.log.emit("sfm", "warn", f"camera path has {jumps} discontinuities between consecutive frames: the model "
                     "may be folded. Recapture (SOP: slower turns) if the twin shows doubled walls.")
    sparse0 = cdir / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)
    best.write(sparse0)

    ctx.log.emit("sfm", "progress", "undistorting to PINHOLE")
    dense = ctx.dense_dir
    if dense.exists():
        shutil.rmtree(dense)
    pycolmap.undistort_images(dense, sparse0, images)
    rec = pycolmap.Reconstruction(dense / "sparse")
    txt = dense / "sparse_txt"
    txt.mkdir(parents=True, exist_ok=True)
    rec.write_text(txt)

    reg = rec.num_reg_images()
    report = {"features": features, "featuresRequested": requested, "matcher": matcher, "loopDetection": loops, "mapper": mapper, "mapperReason": mapper_why, "mapperFallback": fallback, "mapperSeconds": seconds, **extra,
              "trajectoryJumps": jumps, "droppedViews": len(dropped) + len(spikes), "droppedSpikes": spikes, "images": n_images, "registered": reg,
              "registeredRatio": reg / n_images, "models": len(models), "points": rec.num_points3D(),
              "meanReprojErrorPx": float(rec.compute_mean_reprojection_error()),
              "focalPx": float(next(iter(rec.cameras.values())).params[0]), "focalPrior": cam_params}
    if reg / n_images < 0.6:
        ctx.log.emit("sfm", "warn", f"only {reg}/{n_images} frames registered; expect holes (see SOP failure fixes)")
    return report
