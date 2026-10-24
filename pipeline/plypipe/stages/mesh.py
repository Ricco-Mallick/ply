# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""mesh: a textured-looking surface from the same posed frames — dense stereo, fused cloud, poisson mesh.

Splats carry realism but no surface; boxes carry structure but no realism. A dense mesh from the same solve is the
middle: it looks like the footage because its colour comes from the photographs, it is a real mesh (so it exports as
GLB/OBJ and can be cut per object), and it costs no modelling.

Pipeline inside this stage:
  1. a COLMAP dense workspace (images + sparse model) next to the sparse solve
  2. patch-match stereo at a capped resolution, then stereo fusion -> fused.ply (coloured points)
  3. poisson surface reconstruction -> mesh, simplified to keep the file sensible
  4. colour transfer: every mesh vertex takes the colour of the nearest fused point (the fused cloud is coloured,
     the poisson mesh is not), so the mesh renders with the room's own colours
  5. write mesh.ply (vertex colours) and point scene.json at it

Needs CUDA for the stereo step; the rest is CPU. Skipped with a warning if pycolmap has no CUDA.
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from plypipe.util import update_scene
from plypipe.log import Log

MAX_IMAGE_EDGE = 1600      # stereoTakes at this resolution and is then upsampled to the mesh: 2560 buys little here
TARGET_FACES = 900_000     # a phone-friendly mesh; poisson output is usually millions


def _dense_workspace(ctx):
    """COLMAP's dense tools want <workspace>/images and <workspace>/sparse/cameras.bin, not sparse/0/."""
    import pycolmap

    ws = ctx.out / "dense"
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "sparse").mkdir(parents=True)
    # align links data/images to the undistorted frames; when that link is gone (a resumed run, a restored
    # checkpoint) fall back to the dense directory itself, which always holds them
    images = ctx.data_dir / "images"
    if not images.exists():
        images = ctx.dense_dir / "images"
    if not images.exists():
        raise FileNotFoundError(f"{images} missing: the solve must keep its undistorted frames")
    (ws / "images").symlink_to(images.resolve())
    model_dir = ctx.data_dir / "sparse" / "0"
    rec = pycolmap.Reconstruction(str(model_dir))
    rec.write(str(ws / "sparse"))
    return ws


def _colour_transfer(mesh_path, fused_path, out_path, cell=0.04):
    """Give every mesh vertex the colour of the nearest fused point, using a voxel hash instead of a KD-tree."""
    import trimesh  # noqa: F401  (only used when available, for reading the mesh)

    fused = trimesh.load(str(fused_path), process=False)
    pts = np.asarray(fused.vertices, dtype=np.float64)
    cols = np.asarray(fused.colors, dtype=np.uint8)[:, :3] if fused.colors is not None else np.full((len(pts), 3), 180, np.uint8)
    mesh = trimesh.load(str(mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    key = np.floor(pts / cell).astype(np.int64)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    key, pts, cols = key[order], pts[order], cols[order]
    lookup = {}
    for i, k in enumerate(map(tuple, key)):
        lookup.setdefault(k, i)
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    out = np.zeros((len(verts), 3), np.float32)
    pending = []
    for vi, v in enumerate(verts):
        k0 = tuple(np.floor(v / cell).astype(np.int64))
        best, best_d = -1, 1e9
        for o in offsets:
            j = lookup.get((k0[0] + o[0], k0[1] + o[1], k0[2] + o[2]))
            if j is None:
                continue
            d = float(np.sum((pts[j] - v) ** 2))
            if d < best_d:
                best, best_d = j, d
        if best >= 0:
            out[vi] = cols[best] / 255.0
        else:
            pending.append(vi)
    if pending:  # far from any point: fall back to the nearest of a subsample
        sample = np.linspace(0, len(pts) - 1, min(20_000, len(pts))).astype(int)
        for vi in pending:
            d = np.sum((pts[sample] - verts[vi]) ** 2, axis=1)
            out[vi] = cols[sample[int(np.argmin(d))]] / 255.0
    mesh.visual.vertex_colors = np.c_[out, np.ones(len(out))]
    mesh.export(str(out_path))
    return len(verts)


def run_stage(ctx):
    import pycolmap

    # ask torch, not pycolmap: pycolmap.has_cuda changed from a function to a flag between releases, and a stale image
    # served the old call four times. torch.cuda.is_available() is stable across every version we pin.
    import torch

    if not torch.cuda.is_available():
        ctx.log.emit("mesh", "warn", "pycolmap has no CUDA: dense stereo skipped")
        return {"skipped": "no cuda"}

    ws = _dense_workspace(ctx)
    opts = pycolmap.PatchMatchOptions()
    opts.max_image_size = int(ctx.opt("dense_max_edge", MAX_IMAGE_EDGE))
    opts.geom_consistency = True
    ctx.log.emit("mesh", "progress", f"dense stereo at {opts.max_image_size}px")
    pycolmap.patch_match_stereo(str(ws), options=opts)

    fused = ctx.out / "fused.ply"
    ctx.log.emit("mesh", "progress", "fusing depth maps")
    pycolmap.stereo_fusion(str(fused), str(ws), input_type="geometric")

    mesh_raw = ctx.out / "mesh_raw.ply"
    ctx.log.emit("mesh", "progress", "poisson surface")
    popts = pycolmap.PoissonMeshingOptions()
    popts.point_weight = 0.7
    popts.depth = 11
    popts.trim = 8.0
    pycolmap.poisson_meshing(str(fused), str(mesh_raw), popts)

    mesh = ctx.out / "mesh.ply"
    try:
        faces = pycolmap.simplify_mesh(str(mesh_raw), str(mesh), pycolmap.MeshSimplificationOptions()) if False else None
    except Exception:
        faces = None
    # simplify to a sensible size (QEM), then colour it from the fused points
    sops = pycolmap.MeshSimplificationOptions()
    sops.target_num_faces = TARGET_FACES
    try:
        pycolmap.simplify_mesh(str(mesh_raw), str(mesh), sops)
    except Exception as e:
        ctx.log.emit("mesh", "warn", f"simplify failed ({e}); keeping the raw mesh")
        shutil.copyfile(mesh_raw, mesh)
    n = _colour_transfer(mesh, fused, ctx.out / "mesh_colored.ply")
    (ctx.out / "mesh_colored.ply").replace(mesh)
    mesh_raw.unlink(missing_ok=True)

    size_mb = mesh.stat().st_size / 2**20
    update_scene(ctx, {"mesh": {"file": "mesh.ply", "format": "ply-vertex-colour", "vertices": int(n)}})
    return {"mesh": "mesh.ply", "sizeMB": round(size_mb, 2), "vertices": int(n), "fusedMB": round(fused.stat().st_size / 2**20, 1)}
