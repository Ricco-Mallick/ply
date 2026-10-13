"""sfm: frames -> camera poses + sparse points (COLMAP 4.2.1 via pycolmap), undistorted to PINHOLE.

Features: ALIKED (N16ROT) + LightGlue by default, falling back to SIFT if the ONNX models cannot
be loaded. One SIMPLE_RADIAL camera shared by every frame (a phone video is one camera).
Pairs: exhaustive for small captures (<= exhaustive_max frames, best loop closure), sequential
with quadratic overlap otherwise (video order is a strong prior).
Mapper: global SfM (GLOMAP, merged into COLMAP 4.x), keeping the largest model.

Outputs:
    colmap/database.db, colmap/sparse/0          raw model (SIMPLE_RADIAL, binary)
    dense/images, dense/sparse                    undistorted PINHOLE images + binary model
    dense/sparse_txt                              same model as text, read by `align`
"""

from __future__ import annotations

import json

import shutil
from pathlib import Path

from plypipe.util import Ctx


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


def _extract(pycolmap, db: Path, images: Path, features: str, device, camera_params: str | None = None) -> str:
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    if camera_params:
        reader.camera_params = camera_params
    if features == "aliked":
        ext = pycolmap.FeatureExtractionOptions(pycolmap.FeatureExtractorType.ALIKED_N16ROT)
    else:
        ext = pycolmap.FeatureExtractionOptions(pycolmap.FeatureExtractorType.SIFT)
    ext.max_image_size = 1600
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext, device=device)
    return features


def _match(pycolmap, db: Path, features: str, n_images: int, matcher: str, exhaustive_max: int, device) -> str:
    mtype = (pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE if features == "aliked"
             else pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE)
    mopts = pycolmap.FeatureMatchingOptions(mtype)
    if matcher == "auto":
        matcher = "exhaustive" if n_images <= exhaustive_max else "sequential"
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=mopts, device=device)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 20
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, matching_options=mopts, pairing_options=pairing, device=device)
    return matcher


def trajectory_jumps(rec, ratio: float = 8.0) -> int:
    """Video frames are sequential, so consecutive camera centres must move smoothly. A step far
    larger than the median step means the model was stitched from inconsistent pieces, even
    when the reprojection error looks excellent."""
    import numpy as np

    ims = sorted(rec.images.values(), key=lambda im: im.name)
    if len(ims) < 5:
        return 0
    centers = []
    for im in ims:
        if not im.has_pose:
            continue
        pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = pose.rotation.matrix()
        centers.append(-R.T @ pose.translation)
    if len(centers) < 5:
        return 0
    steps = np.linalg.norm(np.diff(np.array(centers), axis=0), axis=1)
    med = float(np.median(steps)) or 1e-12
    return int((steps > ratio * med).sum())


def run_stage(ctx: Ctx) -> dict:
    pycolmap = _pycolmap()
    images = ctx.frames_dir
    n_images = len(list(images.glob("*.jpg")))
    if n_images < 10:
        raise RuntimeError(f"only {n_images} frames: need at least 10 (check the frames stage)")
    device = pycolmap.Device(ctx.opt("device", "auto"))
    cdir = ctx.colmap_dir
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    db = cdir / "database.db"

    features = ctx.opt("features", "aliked")
    cam_params = focal_prior_params(images, float(ctx.opt("focal_35mm"))) if ctx.opt("focal_35mm") else None
    ctx.log.emit("sfm", "progress", f"extracting {features} features", images=n_images, cameraPrior=cam_params)
    try:
        _extract(pycolmap, db, images, features, device, cam_params)
    except Exception as e:  # ONNX model download/runtime missing -> SIFT
        if features != "aliked":
            raise
        ctx.log.emit("sfm", "warn", f"ALIKED failed ({e}); falling back to SIFT")
        db.unlink(missing_ok=True)
        features = "sift"
        _extract(pycolmap, db, images, features, device, cam_params)

    ctx.log.emit("sfm", "progress", "matching")
    # several clips: frames from different clips sit next to each other in the numbering, so sequential pairing
    # would try to join unrelated views and miss the real overlaps; pair exhaustively while that stays affordable
    exhaustive_max = int(ctx.opt("exhaustive_max", 250))
    try:
        source = json.loads((ctx.out / ".done" / "frames.json").read_text()).get("source")
        if source == "videos":
            exhaustive_max = max(exhaustive_max, 700)
        elif source == "ar" and ctx.opt("exhaustive_max") is None:
            # AR keyframes come in walking order, many of them: sequential pairing with quadratic overlap finds the
            # loops too, at a fraction of exhaustive matching's cost (n^2 pairs took 20 min for 146 frames on a T4)
            exhaustive_max = 150
    except (OSError, ValueError):
        pass
    try:
        matcher = _match(pycolmap, db, features, n_images, ctx.opt("matcher", "auto"), exhaustive_max, device)
    except Exception as e:
        if features != "aliked":
            raise
        ctx.log.emit("sfm", "warn", f"LightGlue matching failed ({e}); re-running with SIFT")
        db.unlink(missing_ok=True)
        features = "sift"
        _extract(pycolmap, db, images, features, device, cam_params)
        matcher = _match(pycolmap, db, features, n_images, ctx.opt("matcher", "auto"), exhaustive_max, device)

    mapper = ctx.opt("mapper", "global")
    ctx.log.emit("sfm", "progress", f"{mapper} mapping")
    raw_out = cdir / "sparse_all"
    if mapper == "incremental":
        models = pycolmap.incremental_mapping(db, images, raw_out)
    else:
        models = pycolmap.global_mapping(db, images, raw_out, pycolmap.GlobalPipelineOptions())
    if not models:
        raise RuntimeError(f"{mapper} mapping produced no model: capture lacks overlap or texture (see docs/SOP.md)")
    best_idx, best = max(models.items(), key=lambda kv: kv[1].num_reg_images())
    jumps = trajectory_jumps(best)
    if jumps:
        ctx.log.emit("sfm", "warn", f"camera path has {jumps} discontinuities between consecutive frames: the model "
                     "may be folded. Re-run with --force sfm --mapper incremental, or recapture (SOP: slower turns).")
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
    report = {"features": features, "matcher": matcher, "mapper": mapper, "trajectoryJumps": jumps, "images": n_images, "registered": reg,
              "registeredRatio": reg / n_images, "models": len(models), "points": rec.num_points3D(),
              "meanReprojErrorPx": float(rec.compute_mean_reprojection_error()),
              "focalPx": float(next(iter(rec.cameras.values())).params[0]), "focalPrior": cam_params}
    if reg / n_images < 0.6:
        ctx.log.emit("sfm", "warn", f"only {reg}/{n_images} frames registered; expect holes (see SOP failure fixes)")
    return report
