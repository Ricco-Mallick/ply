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

`--mesh tsdf` (variant 3b, further down) builds mesh.glb and collision.glb from the splats' own depth instead, in export.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
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
    if ctx.opt("mesh") == "tsdf":
        return tsdf_stage(ctx)
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


# ------------------------------------------------------------------ variant 3b: TSDF of the splats' own depth (IMPROVEMENT-PLAN 2.4)
#
# `--mesh tsdf`. The export's dense pass (dense.consistent_points(fuse=True)) hands every kept depth map, pose and colour map to
# Tsdf.integrate as it goes, so nothing is rendered twice. The volume is a hash of 8x8x8-voxel blocks, allocated only where a
# depth map has a surface (within the truncation band), so memory follows the surface and not the room's box: a chapel of
# 25 x 12 x 8 m is about 1,800 m2 of surface, 40 million voxels at 2.5 cm, and a voxel is 7 bytes (tsdf and weight as half
# floats, colour as 3 bytes): about 0.3 GB, which `choose_voxel` keeps under VOXEL_BUDGET by coarsening the voxel (2 to 4 cm).
# Torch only, so the same lines run on CUDA in export and on a CPU in the tests; no scipy and no scikit-image (neither is on the
# worker): the surface is extracted by marching tetrahedra (the Freudenthal split of every cube, whose face diagonals agree
# between neighbours, so the mesh is watertight where the volume is) and cut down by vertex clustering with a quadric error
# position per cell, which keeps a plain wall exactly on its plane at any cell size.

BLOCK = 8
OFFSET = 1 << 20
MESH_FACES, COLLISION_FACES = 300_000, 20_000
VOXEL_RANGE = (0.02, 0.04)
VOXEL_BUDGET = 60e6        # voxels held at once
TRUNC_VOXELS = 3.0         # truncation, in voxels (and at least 6 cm: a 10 cm partition must not be seen through from its other face)
MAX_WEIGHT = 64.0
SPECK_M2 = 0.25            # a connected piece of surface under this (a 50 cm square) is a floater or a flake, not a wall, a pew or a step
MIN_SHARE = 0.1            # a mesh under this share of its target, or under MIN_SPAN of the dense points' extent on any axis, is not shipped
MIN_SPAN = 0.4
# a surface has about two triangles per vertex. The chapel's first counted mesh had 5.7 (298,319 on 52,277 vertices, in 182,297 pieces
# none larger than 261 triangles: a soup of crossing faces, 2026-11-13), which a count and an extent do not catch
MAX_FACES_PER_VERTEX = 3.0
CUBE = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
TETS = [(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)]


def choose_voxel(extent, budget: float = VOXEL_BUDGET, lo: float = VOXEL_RANGE[0], hi: float = VOXEL_RANGE[1]) -> float:
    """The finest voxel in lo..hi (steps of 5 mm) whose estimated volume fits `budget`. `extent` is the scene's bounding box in
    metres. Surface is taken as 1.5 times the box's, and a surface of area A holds about A / (8 v)^2 blocks of 512 voxels, 1.7
    times over for blocks that only the band touches: 25 x 12 x 8 m is 1,788 m2, 61 million voxels at 2 cm and 39 million at 2.5."""
    x, y, z = (float(e) for e in extent)
    area = 1.5 * 2 * (x * y + y * z + x * z)
    v = lo
    while v < hi - 1e-9 and 1.7 * area / (BLOCK * v) ** 2 * BLOCK ** 3 > budget:
        v = round(v + 0.005, 4)
    return v


def _pack(b):
    return ((b[..., 0] + OFFSET) << 42) | ((b[..., 1] + OFFSET) << 21) | (b[..., 2] + OFFSET)


def _unpack(k):
    import torch
    m = (1 << 21) - 1
    return torch.stack([(k >> 42) & m, (k >> 21) & m, k & m], -1) - OFFSET


class Tsdf:
    """Truncated signed distance on a hash of blocks. Distance is positive in front of a surface (free space), negative behind it,
    scaled to -1..1 over the truncation. `integrate` takes one view's filtered depth (0 where unknown) and, optionally, its colours."""

    def __init__(self, voxel: float, device: str = "cpu", trunc: float | None = None):
        import torch
        self.v, self.mu, self.dev = float(voxel), max(trunc or TRUNC_VOXELS * voxel, 0.06), torch.device(device)
        n = BLOCK ** 3
        self.keys = torch.zeros(0, dtype=torch.long, device=self.dev)
        self.tsdf = torch.zeros(0, n, dtype=torch.float16, device=self.dev)
        self.weight = torch.zeros(0, n, dtype=torch.float16, device=self.dev)
        self.colour = torch.zeros(0, n, 3, dtype=torch.uint8, device=self.dev)
        g = torch.arange(BLOCK, device=self.dev)
        self.local = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(n, 3)   # voxel index (lx * 8 + ly) * 8 + lz

    @property
    def blocks(self) -> int:
        return int(len(self.keys))

    @property
    def megabytes(self) -> float:
        return self.blocks * BLOCK ** 3 * 7 / 2 ** 20

    def _allocate(self, keys) -> None:
        import torch
        at = torch.searchsorted(self.keys, keys).clamp(max=max(len(self.keys) - 1, 0))
        new = keys[(self.keys[at] != keys) if len(self.keys) else torch.ones_like(keys, dtype=torch.bool)]
        if not len(new):
            return
        n, cube = len(new), BLOCK ** 3
        allk = torch.cat([self.keys, new])
        order = torch.argsort(allk)
        self.keys = allk[order]
        self.tsdf = torch.cat([self.tsdf, torch.ones(n, cube, dtype=torch.float16, device=self.dev)])[order]
        self.weight = torch.cat([self.weight, torch.zeros(n, cube, dtype=torch.float16, device=self.dev)])[order]
        self.colour = torch.cat([self.colour, torch.zeros(n, cube, 3, dtype=torch.uint8, device=self.dev)])[order]

    def integrate(self, depth, K, R, t, rgb=None, stride: int = 2) -> int:
        """Fuse one view: `depth` (H, W) z-depth with 0 where it is not to be trusted, intrinsics K, world-to-camera R and t, `rgb`
        (H, W, 3) uint8 or None. Returns the blocks the view touched. Allocation looks at every `stride`-th pixel, at the
        surface and a truncation either side; every voxel of a block it touches is then projected into the view."""
        import torch
        dev, mu = self.dev, self.mu
        d = torch.as_tensor(depth, device=dev).float()
        H, W = d.shape
        Rm, tv = (torch.as_tensor(np.asarray(a, np.float32), device=dev) for a in (R, t))
        fx, fy, cx, cy = (float(x) for x in (K[0][0], K[1][1], K[0][2], K[1][2]))
        ys, xs = torch.nonzero(d[::stride, ::stride] > 0, as_tuple=True)
        if not len(ys):
            return 0
        ys, xs = ys * stride, xs * stride
        z = d[ys, xs]
        ray = torch.stack([(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, torch.ones_like(z)], 1)
        world = torch.cat([(ray * (z + o)[:, None] - tv) @ Rm for o in (-mu, 0.0, mu)])   # R^T (p - t)
        keys = torch.unique(_pack(torch.floor(world / (BLOCK * self.v)).long()))
        self._allocate(keys)
        idx = torch.searchsorted(self.keys, keys)
        col = torch.as_tensor(rgb, device=dev) if rgb is not None else None
        for s in range(0, len(idx), 4096):
            ii = idx[s:s + 4096]
            centre = ((_unpack(self.keys[ii])[:, None] * BLOCK + self.local[None] + 0.5) * self.v)            # (n, 512, 3)
            pc = centre @ Rm.T + tv
            zc = pc[..., 2]
            u = torch.floor(fx * pc[..., 0] / zc.clamp(min=1e-6) + cx).long()
            v = torch.floor(fy * pc[..., 1] / zc.clamp(min=1e-6) + cy).long()
            inside = (zc > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v = u.clamp(0, W - 1), v.clamp(0, H - 1)
            dp = d[v, u]
            sdf = dp - zc
            ok = inside & (dp > 0) & (sdf >= -mu)
            w0, t0 = self.weight[ii].float(), self.tsdf[ii].float()
            w1 = w0 + ok.float()
            den = w1.clamp(min=1)
            self.tsdf[ii] = torch.where(ok, (w0 * t0 + (sdf / mu).clamp(-1, 1)) / den, t0).half()
            if col is not None:
                c0 = self.colour[ii].float()
                self.colour[ii] = torch.where(ok[..., None], (w0[..., None] * c0 + col[v, u].float()) / den[..., None], c0).round().clamp(0, 255).to(torch.uint8)
            self.weight[ii] = w1.clamp(max=MAX_WEIGHT).half()
        return len(idx)

    def _neighbours(self):
        """For each block, the index of the block at +1 in each of the eight (dx, dy, dz) of the cube, or -1."""
        import torch
        b = _unpack(self.keys)
        out = {}
        for off in CUBE:
            k = _pack(b + torch.as_tensor(off, device=self.dev))
            at = torch.searchsorted(self.keys, k).clamp(max=len(self.keys) - 1)
            out[tuple(int(o) for o in off)] = torch.where(self.keys[at] == k, at, torch.full_like(at, -1))
        return out

    def extract(self, chunk: int = 1024, min_weight: float = 1.0):
        """The zero level set as (vertices (V, 3) float32, colours (V, 3) uint8, faces (F, 3) int64), normals toward free space. A
        cube is used only when all eight of its corners were seen (weight at least `min_weight`) and the sign changes."""
        import torch
        dev = self.dev
        if not self.blocks:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.int64)
        nb, bxyz = self._neighbours(), _unpack(self.keys)
        g0 = (bxyz.min(0).values * BLOCK)
        cube = torch.as_tensor(CUBE, device=dev)
        parts = []
        for s in range(0, self.blocks, chunk):
            sl = slice(s, min(s + chunk, self.blocks))
            n = sl.stop - sl.start
            T = torch.ones(n, 9, 9, 9, device=dev)
            Wt = torch.zeros(n, 9, 9, 9, device=dev)
            C = torch.zeros(n, 9, 9, 9, 3, device=dev)
            for off, idx in nb.items():
                i = idx[sl]
                has = (i >= 0)
                if not bool(has.any()):
                    continue
                i = i.clamp(min=0)
                dst = tuple(slice(0, 8) if o == 0 else slice(8, 9) for o in off)
                src = tuple(slice(0, 8) if o == 0 else slice(0, 1) for o in off)
                sel = lambda a, k=None: a[i].reshape(n, 8, 8, 8, *(() if k is None else (k,)))[(slice(None),) + src]
                m = has.view(n, 1, 1, 1)
                T[(slice(None),) + dst] = torch.where(m, sel(self.tsdf).float(), T[(slice(None),) + dst])
                Wt[(slice(None),) + dst] = torch.where(m, sel(self.weight).float(), Wt[(slice(None),) + dst])
                C[(slice(None),) + dst] = torch.where(m[..., None], sel(self.colour, 3).float(), C[(slice(None),) + dst])
            corner = lambda a, o: a[:, o[0]:o[0] + 8, o[1]:o[1] + 8, o[2]:o[2] + 8]
            tv = torch.stack([corner(T, o) for o in CUBE.tolist()], -1)
            seen = torch.stack([corner(Wt, o) for o in CUBE.tolist()], -1).min(-1).values >= min_weight
            hit = seen & (tv.min(-1).values < 0) & (tv.max(-1).values >= 0)
            bi, i, j, k = torch.nonzero(hit, as_tuple=True)
            if not len(bi):
                continue
            base = torch.stack([i, j, k], 1)
            nodes = base[:, None] + cube[None]                                   # (m, 8, 3) voxel offsets inside the padded block
            val = tv[bi, i, j, k]                                                # (m, 8)
            rgb = torch.stack([C[bi, nodes[:, c, 0], nodes[:, c, 1], nodes[:, c, 2]] for c in range(8)], 1)   # (m, 8, 3)
            glob = bxyz[sl][bi][:, None] * BLOCK + nodes - g0                    # (m, 8, 3) voxel coordinates from the volume's corner
            pos = (glob + g0 + 0.5) * self.v
            nid = (glob[..., 0] << 32) | (glob[..., 1] << 16) | glob[..., 2]    # 16 bits an axis: 65,000 voxels, 1.3 km at 2 cm
            tri = []
            for tet in TETS:
                tri.append(_tet(val[:, tet], rgb[:, tet], pos[:, tet], nid[:, tet], glob[:, tet]))
            keys = torch.cat([a[0] for a in tri if a is not None])
            if not len(keys):
                continue
            vpos, vcol = torch.cat([a[1] for a in tri if a is not None]), torch.cat([a[2] for a in tri if a is not None])
            flat = keys.reshape(-1)
            uk, inv = torch.unique(flat, return_inverse=True)
            p, c = torch.zeros(len(uk), 3, device=dev), torch.zeros(len(uk), 3, device=dev)
            p[inv], c[inv] = vpos.reshape(-1, 3), vcol.reshape(-1, 3)
            parts.append((uk, p, c, inv.reshape(-1, 3)))
        if not parts:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.int64)
        shift = np.cumsum([0] + [len(p[0]) for p in parts])
        keys, pos, col = (torch.cat([p[i] for p in parts]) for i in range(3))
        faces = torch.cat([p[3] + int(o) for p, o in zip(parts, shift)])
        uk, inv = torch.unique(keys, return_inverse=True)
        vp, vc = torch.zeros(len(uk), 3, device=dev), torch.zeros(len(uk), 3, device=dev)
        vp[inv], vc[inv] = pos, col
        faces = inv[faces]
        return vp.cpu().numpy().astype(np.float32), vc.round().clamp(0, 255).to(torch.uint8).cpu().numpy(), faces.cpu().numpy().astype(np.int64)


def _edge(a, b, val, rgb, pos, nid, glob):
    """The surface crossing on the edges from corner a to corner b of each tetrahedron (index arrays): (key, position, colour).
    Always computed from the lower node to the higher, so that the two tetrahedra that share an edge agree to the last bit."""
    import torch
    r = torch.arange(len(a), device=a.device)
    na, nb_ = nid[r, a], nid[r, b]
    swap = na > nb_
    lo, hi = torch.where(swap, b, a), torch.where(swap, a, b)
    d = glob[r, hi] - glob[r, lo]
    key = nid[r, lo] * 27 + ((d[:, 0] + 1) * 3 + d[:, 1] + 1) * 3 + d[:, 2] + 1
    v0, v1 = val[r, lo], val[r, hi]
    f = (v0 / (v0 - v1)).clamp(0, 1)[:, None]
    return key, pos[r, lo] + (pos[r, hi] - pos[r, lo]) * f, rgb[r, lo] + (rgb[r, hi] - rgb[r, lo]) * f


def _tet(val, rgb, pos, nid, glob):
    """Triangles of one tetrahedron per row: (keys (T, 3), positions (T, 3, 3), colours (T, 3, 3)) or None. One corner inside
    (negative) or one outside is a triangle, two and two a quad cut in two; each faces from the inside corners to the outside."""
    import torch
    inside = val < 0
    cnt = inside.sum(1)
    keys, vpos, vcol = [], [], []

    def verts(pairs, rows):
        out = [_edge(a, b, val[rows], rgb[rows], pos[rows], nid[rows], glob[rows]) for a, b in pairs]
        return [torch.stack([o[i] for o in out], 1) for i in range(3)]   # keys (m, q), pos (m, q, 3), col (m, q, 3)

    rows = torch.nonzero((cnt == 1) | (cnt == 3)).reshape(-1)
    if len(rows):
        ins = inside[rows]
        minor = torch.where((cnt[rows] == 1)[:, None], ins, ~ins)
        mi = minor.float().argmax(1)
        others = [(mi + 1 + k) % 4 for k in range(3)]
        k_, p_, c_ = verts([(mi, o) for o in others], rows)
        pr = pos[rows]
        r = torch.arange(len(rows), device=val.device)
        centroid = sum(pr[r, o] for o in others) / 3
        toward = torch.where((cnt[rows] == 1)[:, None], centroid - pr[r, mi], pr[r, mi] - centroid)
        flip = (torch.cross(p_[:, 1] - p_[:, 0], p_[:, 2] - p_[:, 0], dim=1) * toward).sum(1) < 0
        order = torch.where(flip[:, None], torch.tensor([0, 2, 1], device=val.device), torch.tensor([0, 1, 2], device=val.device))
        keys.append(torch.gather(k_, 1, order))
        vpos.append(torch.gather(p_, 1, order[..., None].expand(-1, -1, 3)))
        vcol.append(torch.gather(c_, 1, order[..., None].expand(-1, -1, 3)))
    rows = torch.nonzero(cnt == 2).reshape(-1)
    if len(rows):
        order4 = torch.argsort((~inside[rows]).float(), dim=1, stable=True)
        i, j, k, l = (order4[:, q] for q in range(4))
        k_, p_, c_ = verts([(i, k), (i, l), (j, l), (j, k)], rows)
        pr = pos[rows]
        r = torch.arange(len(rows), device=val.device)
        toward = (pr[r, k] + pr[r, l] - pr[r, i] - pr[r, j]) / 2
        flip = (torch.cross(p_[:, 1] - p_[:, 0], p_[:, 2] - p_[:, 0], dim=1) * toward).sum(1) < 0
        order = torch.where(flip[:, None], torch.tensor([0, 3, 2, 1], device=val.device), torch.tensor([0, 1, 2, 3], device=val.device))
        k_ = torch.gather(k_, 1, order)
        p_ = torch.gather(p_, 1, order[..., None].expand(-1, -1, 3))
        c_ = torch.gather(c_, 1, order[..., None].expand(-1, -1, 3))
        for tri in ([0, 1, 2], [0, 2, 3]):
            keys.append(k_[:, tri])
            vpos.append(p_[:, tri])
            vcol.append(c_[:, tri])
    if not keys:
        return None
    return torch.cat(keys), torch.cat(vpos), torch.cat(vcol)


def decimate(vertices: np.ndarray, faces: np.ndarray, colours: np.ndarray | None, target: int, iters: int = 16):
    """Vertex clustering to about `target` triangles (never more than 5 % over, unless the volume's own cell is already coarser):
    the vertices of a grid cell become one, placed where the summed squared distance to the planes of its triangles is least,
    pulled toward the cell's mean vertex (so an unconstrained direction stays put) and kept within the cell grown by a quarter.
    A plain wall's triangles all lie on one plane, so the vertex stays on it: the wall is as flat at 20k triangles as at 2M.
    The cell is found by bisection on its size. Returns (vertices, faces, colours)."""
    v, f = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    if len(f) <= target:
        return vertices, faces, colours
    origin = v.min(0)
    span = float((v.max(0) - origin).max()) or 1.0
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])   # area-weighted face normals, summed at the vertices, then binned by
    vn = np.stack([sum(np.bincount(f[:, k], weights=fn[:, a], minlength=len(v)) for k in range(3)) for a in range(3)], 1)   # their dominant
    side = np.argmax(np.abs(vn), 1) * 2 + (vn[np.arange(len(v)), np.argmax(np.abs(vn), 1)] < 0)    # axis and sign, so that two sheets a
    # few centimetres apart (a thin partition) are never one vertex: a collision mesh must not put the wall between its faces
    vw = sum(np.bincount(f[:, k], weights=np.linalg.norm(fn, axis=1), minlength=len(v)) for k in range(3))

    def cluster(cell):
        q = np.floor((v - origin) / cell).astype(np.int64)
        dims = q.max(0) + 1
        _, cid = np.unique((q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2], return_inverse=True)
        # Noise scatters a flat patch's vertices over all six sides; a side with under a quarter of its cell's weight joins the
        # heaviest, or one cell becomes six vertices and the faces between them a pile on one plane (19 k vertices under 299 k
        # triangles on a 1 cm noised chapel). A second sheet of its own weight, the thin partition's, stays.
        w = np.bincount(cid * 6 + side, weights=vw, minlength=(int(cid.max()) + 1) * 6).reshape(-1, 6)
        eff = np.where(w[cid, side] >= 0.25 * w.max(1)[cid], side, w.argmax(1)[cid])
        _, inv = np.unique(cid * 6 + eff, return_inverse=True)
        cf = inv[f]
        return inv, cf, (cf[:, 0] != cf[:, 1]) & (cf[:, 1] != cf[:, 2]) & (cf[:, 0] != cf[:, 2])

    def distinct(cf, keep):
        """Triangles left once the same corner triple, made by two folded sheets, is counted once: what the output will hold. Counting
        the kept faces instead overshoots wherever noise spreads a vertex's side over all six, by a fifth of all faces on a hall
        noised by a third of a voxel, and then no cell reached the target. A 64-bit hash of the sorted triple: a collision is negligible at these sizes."""
        t = cf[keep].astype(np.uint64)
        a, c = t.min(1), t.max(1)
        return int(len(np.unique(a * np.uint64(0x9E3779B97F4A7C15) + (t.sum(1) - a - c) * np.uint64(0xC2B2AE3D27D4EB4F) + c * np.uint64(0x165667B19E3779F9))))

    lo, hi = span / 4096, span / 2
    cell = min(max(math.sqrt(np.linalg.norm(fn, axis=1).sum() / target), lo), hi)   # about target / 2 cells of the surface's area: where the search starts
    for _ in range(iters):
        got = distinct(*cluster(cell)[1:])
        if got > target:
            lo = cell
        else:
            hi = cell
            if got >= 0.99 * target:
                break
        cell = math.sqrt(lo * hi) if lo > span / 4096 and hi < span / 2 else min(max(cell * (1.5 if got > target else 1 / 1.5), span / 4096), span / 2)
    cell = hi
    inv, cf, keep = cluster(cell)
    nc = int(inv.max()) + 1
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(area, 1e-12)[:, None]
    p4 = np.c_[n, -(n * a).sum(1)]
    w = area / 2
    Q = np.zeros((nc, 4, 4))
    for r in range(4):
        for s in range(r, 4):
            q = w * p4[:, r] * p4[:, s]
            acc = sum(np.bincount(cf[:, k], weights=q, minlength=nc) for k in range(3))
            Q[:, r, s] = Q[:, s, r] = acc
    count = np.bincount(inv, minlength=nc).astype(np.float64)
    mean = np.stack([np.bincount(inv, weights=v[:, k], minlength=nc) for k in range(3)], 1) / np.maximum(count, 1)[:, None]
    A, bvec = Q[:, :3, :3], Q[:, :3, 3]
    eps = 1e-2 * np.maximum(np.trace(A, axis1=1, axis2=2) / 3, 1e-9)
    x = np.linalg.solve(A + eps[:, None, None] * np.eye(3), (-bvec + eps[:, None] * mean)[..., None])[..., 0]
    box_lo = origin + np.floor((mean - origin) / cell) * cell - 0.25 * cell
    x = np.clip(x, box_lo, box_lo + 1.5 * cell)
    x = np.where(np.isfinite(x).all(1)[:, None], x, mean)
    cf = cf[keep]
    used = np.unique(cf)
    remap = np.full(nc, -1, np.int64)
    remap[used] = np.arange(len(used))
    out_f = remap[cf]
    key = np.sort(out_f, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)   # the same triangle twice, from two folded sheets
    out_f = out_f[np.sort(first)]
    out_c = None
    if colours is not None:
        cs = np.stack([np.bincount(inv, weights=colours[:, k].astype(np.float64), minlength=nc) for k in range(3)], 1) / np.maximum(count, 1)[:, None]
        out_c = cs[used].round().astype(np.uint8)
    return x[used].astype(np.float32), out_f, out_c


def components(n: int, faces) -> np.ndarray:
    """Connected-component label of each of `n` vertices over the edges of `faces`. Numpy only (the worker's image has no scipy):
    each round hooks the larger root of every edge under the smaller, then jumps pointers, so rounds are few and edges inside
    one component drop out as they are found."""
    a = np.concatenate([faces[:, 0], faces[:, 1]]).astype(np.int32)
    b = np.concatenate([faces[:, 1], faces[:, 2]]).astype(np.int32)
    label = np.arange(n, dtype=np.int32)
    while len(a):
        la, lb = label[a], label[b]
        open_ = la != lb
        a, b, la, lb = a[open_], b[open_], la[open_], lb[open_]
        if not len(a):
            break
        label[np.maximum(la, lb)] = np.minimum(la, lb)   # a root points at a smaller label of its own component, whichever edge wins
        while True:
            up = label[label]
            if (up == label).all():
                break
            label = up
    return label


def drop_specks(vertices, faces, colours, min_area: float = SPECK_M2):
    """Remove connected pieces of surface under `min_area` m2: floaters and flakes the depth confirmed. The size is physical so
    that it means the same at 2 cm and at 4 cm, and the same on a fragmented real fusion as on a clean synthetic room, where a
    share of the triangles (0.2 %, which was 4 m2 and took a third of the chapel) did not. Returns (vertices, faces, colours,
    stats) with the area kept and dropped in m2 and the pieces dropped."""
    if not len(faces):
        return vertices, faces, colours, {"keptM2": 0.0, "droppedM2": 0.0, "droppedPieces": 0}
    n = len(vertices)
    v = np.asarray(vertices, np.float64)
    area = np.linalg.norm(np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]), axis=1) / 2
    lab = components(n, faces)
    piece = lab[faces[:, 0]]
    total = np.bincount(piece, weights=area, minlength=n)
    keep_f = total[piece] >= min_area
    stats = {"keptM2": round(float(area[keep_f].sum()), 1), "droppedM2": round(float(area[~keep_f].sum()), 1),
             "droppedPieces": int(len(np.unique(piece[~keep_f])))}
    used = np.unique(faces[keep_f])
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces[keep_f]], None if colours is None else colours[used], stats


def write_glb(path, vertices, faces, colours=None) -> int:
    import trimesh
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if colours is not None:
        m.visual.vertex_colors = np.c_[colours, np.full(len(colours), 255, np.uint8)]
    m.export(str(path), file_type="glb")
    return int(Path(path).stat().st_size)


class Fusion:
    """What export and the mesh stage hold while the dense pass runs: `begin` sizes the voxel from the cameras, `add` takes each
    dense.Points (made with fuse=True), `finish` writes mesh.glb and collision.glb and returns the report."""

    def __init__(self, device: str = "cpu", voxel: float | None = None, faces: int = MESH_FACES, collision: int = COLLISION_FACES, margin: float = 4.0):
        self.device, self.fixed, self.faces, self.collision, self.margin = device, voxel, faces, collision, margin
        self.tsdf: Tsdf | None = None
        self.views = 0
        self.keep_raw = False  # export sets it for --mesh-keep-raw: the extraction is saved as mesh_raw.npz before anything is dropped
        self.t0 = time.time()

    def begin(self, views) -> None:
        c = np.array([v.center for v in views])
        extent = (c.max(0) - c.min(0)) + 2 * self.margin if len(c) else np.array([8.0, 3.0, 8.0])
        self.tsdf = Tsdf(self.fixed or choose_voxel(extent), self.device)

    def add(self, p) -> None:
        if self.tsdf is None:
            self.begin([p.cam])
        self.tsdf.integrate(p.depth, p.cam.K, p.cam.R, p.cam.t, p.rgb)
        self.views += 1

    def finish(self, out_dir: Path, reference=None) -> dict:
        """`reference` is `dense_reference(...)` of the dense points: with it a mesh that covers far less than they do is refused."""
        t1 = time.time()
        ts = self.tsdf
        if ts is None or not ts.blocks:
            return {"built": False, "why": "no depth was fused"}
        v, c, f = ts.extract()
        raw = len(f)
        if self.keep_raw:
            np.savez(Path(out_dir) / "mesh_raw.npz", vertices=np.asarray(v, np.float32), faces=np.asarray(f, np.int32), colours=np.asarray(c, np.uint8))
        v, f, c, specks = drop_specks(v, f, c)
        mv, mf, mc = decimate(v, f, c, self.faces)
        cv, cf, _ = decimate(mv, mf, None, self.collision)
        out = {"voxelCm": round(ts.v * 100, 1), "truncationCm": round(ts.mu * 100, 1), "views": self.views, "blocks": ts.blocks,
               "volumeMB": round(ts.megabytes, 1), "rawTriangles": raw, "specksDropped": raw - len(f),
               "specks": {**specks, "underM2": SPECK_M2}, "triangles": int(len(mf)), "collisionTriangles": int(len(cf)), **({"denseSurfaceM2": reference["surfaceM2"]} if reference else {})}
        why = refusal(len(f), mv, mf, cv, cf, self.faces, self.collision, reference)
        if why:
            return {"built": False, "why": why, **out, "fuseSeconds": round(t1 - self.t0, 1), "meshSeconds": round(time.time() - t1, 1)}
        out_dir = Path(out_dir)
        mb = write_glb(out_dir / "mesh.glb", mv, mf, mc)
        cb = write_glb(out_dir / "collision.glb", cv, cf)
        return {"built": True, **out, "meshMB": round(mb / 2 ** 20, 2), "collisionMB": round(cb / 2 ** 20, 2), "fuseSeconds": round(t1 - self.t0, 1),
                "meshSeconds": round(time.time() - t1, 1)}


def dense_reference(points_path) -> dict | None:
    """Bounds of the dense points and an upper bound of their surface (one voxel face each: the points are one per voxel, and the
    depth's thickness makes it generous), the yardstick `refusal` holds a mesh against. None without the file."""
    p = Path(points_path)
    if not p.exists():
        return None
    z = np.load(p)
    pts = z["points"]
    return {"min": pts.min(0), "max": pts.max(0), "surfaceM2": round(float(len(pts)) * float(z["voxel"]) ** 2, 0)}


def refusal(have: int, mv, mf, cv, cf, faces: int, collision: int, reference=None) -> str | None:
    """Why a decimated mesh must not ship, or None. A mesh of under MIN_SHARE of its target (a smaller one than the target is
    only fine when that is all there was: `have` triangles went in), or whose bounds are under MIN_SPAN of the dense points' on
    any axis, is a failed run, not a small room: the chapel's came out at 176 triangles of a 300,000 target."""
    if not have or not len(mf):
        return "no surface was left after the specks were dropped"
    for name, got, target in (("mesh", len(mf), faces), ("collision mesh", len(cf), collision)):
        want = min(target, have)
        if got < MIN_SHARE * want:
            return f"the {name} has {got} triangles, under {MIN_SHARE:.0%} of its {want}"
    for name, vv, ff in (("mesh", mv, mf), ("collision mesh", cv, cf)):
        per = len(ff) / max(1, len(np.unique(ff)))
        if per > MAX_FACES_PER_VERTEX:
            return f"the {name} has {per:.1f} triangles per vertex, over {MAX_FACES_PER_VERTEX:.0f}: crossing faces, not a surface"
    if reference is not None and len(mv):
        span = (mv.max(0) - mv.min(0)) / np.maximum(reference["max"] - reference["min"], 1e-6)
        if span.min() < MIN_SPAN:
            return f"the mesh covers {span.min():.0%} of the dense points' extent on its worst axis, under {MIN_SPAN:.0%}"
    return None


def tsdf_stage(ctx) -> dict:
    """`--mesh tsdf` as a stage of its own: the export already built mesh.glb when its dense pass ran with the option (the
    usual case, and the only one that does not render twice), so that is reported and kept; otherwise the stage renders the
    cleaned twin itself, which needs a GPU."""
    from plypipe import dense

    glb = ctx.out / "mesh.glb"
    if glb.exists():
        return {"mesh": glb.name, "from": "export", "sizeMB": round(glb.stat().st_size / 2 ** 20, 2)}
    import torch
    ply = next((p for p in (ctx.out / "scene_clean.ply", ctx.out / "scene.ply") if p.exists()), None)
    cams = dense.training_views(ctx.data_dir, int(ctx.opt("holdout", 30) or 0), 5, 0.5)
    rendered = dense.render_views(ply, cams, not ctx.opt("no_antialias"), colour=True) if ply and cams else None
    if rendered is None or not torch.cuda.is_available():
        ctx.log.emit("mesh", "warn", "tsdf mesh skipped: needs a GPU with gsplat, a trained twin and PINHOLE cameras")
        return {"skipped": "needs a GPU with gsplat, a trained twin and PINHOLE cameras"}
    return finish_job(ctx, Fusion("cuda"), cams, rendered)


def finish_job(ctx, fusion: Fusion, cams, rendered) -> dict:
    from plypipe import dense

    fusion.begin(cams)
    for p in dense.consistent_points(rendered, fuse=True):
        fusion.add(p)
    out = record(ctx, fusion.finish(ctx.out, dense_reference(ctx.out / dense.FILE)))
    if not out.get("built"):
        ctx.log.emit("mesh", "warn", f"tsdf mesh not shipped, no mesh.glb or collision.glb written: {out.get('why')}")
    return out


def record(ctx, out: dict) -> dict:
    """Point scene.json at the files a finished fusion wrote."""
    if out.get("built"):
        update_scene(ctx, {"mesh": {"file": "mesh.glb", "collision": "collision.glb", "format": "glb-vertex-colour", "source": "tsdf",
                                    "triangles": out["triangles"], "collisionTriangles": out["collisionTriangles"]}})
    return out
