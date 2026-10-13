"""export: scene.ply -> scene.spz (full) + scene_lite.spz (SH0 only, for phones).

Uses @playcanvas/splat-transform (pinned) through npx. The PLY is already in the scene frame
(training ran on the aligned model), so no axis change is applied. Gaussians with NaN/Inf are
removed and everything outside the room box (scene bounds + 1 m, y in [-0.5, top + 0.5]) is
cropped, which drops most floaters for free.

Cleanup (on unless --no-clean): near-invisible Gaussians (opacity < 3 %) and oversized blobs (any axis > 1 m) are
dropped with value filters, then splat-transform's floater filter removes Gaussians that don't contribute to any solid
voxel. The floater filter needs WebGPU; without it the stage logs a warning and exports the value-filtered model.
The phone file is SH0 and, above LITE_MAX Gaussians, decimated (phones crash on multi-million-splat scenes).

If a PLY from elsewhere (trained on the unaligned model) is given via --ply, pass --apply-align
and the stage applies scene.json's transform with splat-transform's translate/rotate/scale.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from plypipe.util import Ctx, need, read_scene, run, splat_transform_cmd, update_scene

SPLAT_TRANSFORM = "@playcanvas/splat-transform@3.10.1"
LITE_MAX = 800_000
# splat-transform compares activated values: opacity 0..1, scales in metres
CLEAN = ["-V", "opacity,gt,0.03", "-V", "scale_0,lt,1", "-V", "scale_1,lt,1", "-V", "scale_2,lt,1"]


def ply_count(ply: Path) -> int:
    """Vertex count from a PLY header."""
    with ply.open("rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


# splat-transform runs its actions (-r/-s/-t/-B) in PlayCanvas's engine frame, which is the PLY
# frame turned 180 deg about Z: engine = D @ ply with D = diag(-1, -1, 1). Measured against the
# real tool in tests/test_export.py. Its SPZ writer stores positions unconverted (spz
# CoordinateSystem.UNSPECIFIED), so the SPZ keeps the PLY's scene frame (+Y up, three.js-style).
D = np.diag([-1.0, -1.0, 1.0])


def _euler_for_splat_transform(R: np.ndarray) -> tuple[float, float, float]:
    """Degrees (a, b, c) such that `-r a,b,c` applies R = Rz(c) @ Ry(b) @ Rx(a) in the engine frame."""
    b = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) < 0.99999:
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: fold everything into a
        a = math.atan2(-R[1, 2], R[1, 1])
        c = 0.0
    return math.degrees(a), math.degrees(b), math.degrees(c)


def align_actions(T: np.ndarray) -> list[str]:
    """splat-transform actions applying p' = s R p + t to PLY-frame data (rotate, scale, translate)."""
    sR = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(sR)))
    R_e = D @ (sR / s) @ D
    t_e = D @ T[:3, 3]
    ax, ay, az = _euler_for_splat_transform(R_e)
    return ["-r", f"{ax:.6f},{ay:.6f},{az:.6f}", "-s", f"{s:.8f}", "-t", f"{t_e[0]:.6f},{t_e[1]:.6f},{t_e[2]:.6f}"]


def crop_box(scene: dict) -> str | None:
    """Room box (scene bounds + 1 m sideways, y from -0.5 to top + 0.5) in splat-transform's frame."""
    b = scene.get("bounds")
    if not b:
        return None
    mn, mx = b["min"], b["max"]
    lo = np.array([mn[0] - 1.0, -0.5, mn[2] - 1.0])
    hi = np.array([mx[0] + 1.0, max(mx[1], 2.5) + 0.5, mx[2] + 1.0])
    e_lo, e_hi = np.minimum(D @ lo, D @ hi), np.maximum(D @ lo, D @ hi)
    return ",".join(f"{v:.3f}" for v in (*e_lo, *e_hi))


def run_stage(ctx: Ctx) -> dict:
    st = splat_transform_cmd()
    if st[0] == "npx":
        need("npx", "Install Node.js >= 20 (splat-transform runs through npx).")
    ply = Path(ctx.opt("ply") or ctx.out / "scene.ply")
    if not ply.exists():
        raise FileNotFoundError(f"{ply} missing: run the train stage or pass --ply")
    scene = read_scene(ctx)
    # SPZ v3 (gzip): the web viewer's Spark 2.3.1 rejects v4, splat-transform's default ("Unsupported SPZ version: 4")
    base = [*st, "-w", "--spz-version", "3", str(ply)]
    if ctx.opt("apply_align"):
        base += align_actions(np.array(scene["transform"]))
    base += ["-N"]
    box = crop_box(scene) if not ctx.opt("no_crop") else None
    if box:
        base += ["-B", box]
    clean = not ctx.opt("no_clean")
    if clean:
        base += CLEAN
    full = ctx.out / "scene.spz"
    lite = ctx.out / "scene_lite.spz"
    cleaned = ctx.out / "scene_clean.ply"
    # one cleaned PLY first (floaters filter when WebGPU is there), then both web files from it
    floaters = False
    if clean:
        try:
            run(base + ["-F", str(cleaned)], ctx, "export")
            floaters = True
        except Exception as e:  # no WebGPU adapter in this container: keep the value filters only
            ctx.log.emit("export", "warn", f"floater filter unavailable ({str(e)[-160:]}); exporting without it")
            run(base + [str(cleaned)], ctx, "export")
    else:
        run(base + [str(cleaned)], ctx, "export")
    src = [*st, "-w", "--spz-version", "3", str(cleaned)]
    run(src + [str(full)], ctx, "export")
    n = ply_count(cleaned)
    if n > LITE_MAX:  # decimation must end in a .ply: decimate, then convert
        lite_ply = ctx.out / "scene_lite.ply"
        run([*st, "-w", str(cleaned), "-H", "0", "-d", str(LITE_MAX), str(lite_ply)], ctx, "export")
        run([*st, "-w", "--spz-version", "3", str(lite_ply), str(lite)], ctx, "export")
        lite_ply.unlink(missing_ok=True)
    else:
        run(src + ["-H", "0", str(lite)], ctx, "export")
    update_scene(ctx, {"splat": {"file": "scene.spz", "lite": "scene_lite.spz", "format": "spz"}})
    return {"spz": full.name, "spzMB": round(full.stat().st_size / 2**20, 2), "lite": lite.name, "liteMB": round(lite.stat().st_size / 2**20, 2),
            "cropBox": box, "gaussiansIn": ply_count(ply), "gaussiansOut": n, "floaterFilter": floaters, "liteGaussians": min(n, LITE_MAX)}
