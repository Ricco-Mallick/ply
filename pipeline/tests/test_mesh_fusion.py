"""The TSDF mesh (stages/mesh.py, IMPROVEMENT-PLAN 2.4, variant 3b) on the synthetic two rooms of synth.py: ray-cast depth stands in
for the splats' rendered depth, goes through dense.consistent_points exactly as export does, and is fused on a CPU by the same
torch code that runs on CUDA. The rooms are 8.1 x 4 x 2.8 m with a 10 cm partition and a 0.8 x 2 m door."""
import numpy as np
import pytest

pytest.importorskip("torch")  # the fusion is torch on a CPU here and on CUDA in export; CI has no torch
import trimesh  # noqa: E402

from plypipe import dense  # noqa: E402
from plypipe.stages import mesh  # noqa: E402

from synth import NAV_DOOR, NAV_PARTITION, NAV_ROOMS, depth_views, look, nav_boxes, nav_walk  # noqa: E402

COLOUR = (200, 120, 40)


@pytest.fixture(scope="module")
def two_rooms():
    centers, _ = nav_walk()
    views = [(c, look(k * np.pi / 2 + 0.4 + 0.3 * i, 0.1 if k % 2 else -0.1)) for i, c in enumerate(centers[::12]) for k in range(4)]
    vs = depth_views(nav_boxes(), np.array([c for c, _ in views]), np.array([r for _, r in views]))
    for v in vs:
        v.rgb = np.broadcast_to(np.array(COLOUR, np.uint8), (v.height, v.width, 3)).copy()
    return vs


@pytest.fixture(scope="module")
def fused(two_rooms):
    fusion = mesh.Fusion(voxel=0.025)
    fusion.begin(two_rooms)
    for p in dense.consistent_points(two_rooms, fuse=True):
        fusion.add(p)
    return fusion


@pytest.fixture(scope="module")
def built(fused, tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh")
    report = fused.finish(out)
    return out, report, trimesh.load(out / "mesh.glb", force="mesh"), trimesh.load(out / "collision.glb", force="mesh")


def test_the_dense_pass_hands_the_fusion_exactly_the_depth_its_points_came_from(two_rooms):
    got = list(dense.consistent_points(two_rooms[:12], fuse=True))
    assert all(p.cam is not None and int((np.asarray(p.depth) > 0).sum()) == len(p.points) and p.rgb is not None for p in got)
    plain = list(dense.consistent_points(two_rooms[:12]))
    assert all(a.depth is None and a.cam is None and np.array_equal(a.points, b.points) for a, b in zip(plain, got))


def test_walls_are_within_two_centimetres_and_the_floor_is_at_y_zero(built):
    _, _, m, _ = built
    v = np.asarray(m.vertices)
    W, D, H = NAV_ROOMS
    for sel, axis, plane in [((v[:, 2] < 0.1) & (v[:, 0] > 0.3) & (v[:, 0] < 3.7) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 2, 0.0),
                             ((v[:, 2] > D - 0.1) & (v[:, 0] > 0.3) & (v[:, 0] < 3.7) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 2, D),
                             ((v[:, 0] < 0.1) & (v[:, 2] > 0.3) & (v[:, 2] < D - 0.3) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 0, 0.0),
                             ((v[:, 0] > W - 0.1) & (v[:, 2] > 0.3) & (v[:, 2] < D - 0.3) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 0, W),
                             ((v[:, 1] < 0.1) & (v[:, 0] > 0.3) & (v[:, 0] < 6.5) & ((v[:, 0] < 3.9) | (v[:, 0] > 4.2)) & (v[:, 2] > 0.3) & (v[:, 2] < D - 0.3), 1, 0.0)]:
        assert sel.sum() > 200
        assert np.abs(v[sel, axis] - plane).max() < 0.02, (axis, plane, np.abs(v[sel, axis] - plane).max())
    assert abs(float(v[:, 1].min())) < 0.02 and abs(float(v[:, 1].max()) - H) < 0.02


def test_the_door_is_an_opening_in_the_mesh_and_the_partition_beside_it_is_not(built):
    _, _, m, _ = built
    v = np.asarray(m.vertices)
    (p0, p1), (z0, z1, top) = NAV_PARTITION, NAV_DOOR[:2] + (NAV_DOOR[2],)
    on_plane = (v[:, 0] > p0 - 0.03) & (v[:, 0] < p1 + 0.03)
    gap = on_plane & (v[:, 2] > z0 + 0.08) & (v[:, 2] < z1 - 0.08) & (v[:, 1] > 0.1) & (v[:, 1] < top - 0.1)
    solid = on_plane & (v[:, 2] > 0.1) & (v[:, 2] < z0 - 0.2) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5)
    assert gap.sum() == 0 and solid.sum() > 100


def test_the_collision_mesh_is_within_its_budget_and_within_five_centimetres_of_the_mesh(built):
    out, report, m, c = built
    assert report["built"] and len(m.faces) <= mesh.MESH_FACES * 1.05 and len(c.faces) <= mesh.COLLISION_FACES * 1.05, report
    assert len(c.faces) < len(m.faces) or len(m.faces) <= mesh.COLLISION_FACES * 1.05
    from scipy.spatial import cKDTree
    a, b = m.sample(600_000), c.sample(600_000)
    tree = cKDTree(a)
    # a collision triangle may bridge the hole of a patch no camera saw: 6 cm at most, on 0.05 % of the surface
    near = [tree.query(x)[0] for x in (c.vertices, b)]
    assert all(np.percentile(d, 99.9) < 0.05 and d.max() < 0.07 for d in near), [d.max() for d in near]
    assert (out / "mesh.glb").stat().st_size > 0 and (out / "collision.glb").stat().st_size > 0


def test_vertex_colours_are_the_fused_colours_and_normals_face_the_room(built):
    _, _, m, _ = built
    col = np.asarray(m.visual.vertex_colors)[:, :3].astype(int)
    assert np.abs(col - np.array(COLOUR)).max() <= 1
    near = m.triangles_center[:, 2] < 0.05            # the wall z = 0: its normals point into the room, +z
    assert near.sum() > 20 and float(m.face_normals[near, 2].mean()) > 0.9


def test_a_plane_seen_by_one_camera_is_meshed_at_its_distance_and_decimation_keeps_it_flat():
    from plypipe.stages.mesh import Tsdf, decimate
    K = np.array([[120.0, 0, 80], [0, 120.0, 60], [0, 0, 1]])
    depth = np.full((120, 160), 3.0, np.float32)
    ts = Tsdf(0.03)
    ts.integrate(depth, K, np.eye(3), np.zeros(3))
    v, c, f = ts.extract()
    assert len(f) > 1000 and np.abs(v[:, 2] - 3.0).max() < 0.01 and (np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2] < 0).mean() > 0.99
    dv, df, _ = decimate(v, f, None, 40)
    assert len(df) <= 42 and np.abs(dv[:, 2] - 3.0).max() < 0.01


def test_the_voxel_is_chosen_by_the_size_of_the_scene_and_a_chapel_fits_in_memory():
    assert mesh.choose_voxel((8, 3, 4)) == 0.02
    chapel = mesh.choose_voxel((25, 12, 8))
    assert 0.02 <= chapel <= 0.03
    assert mesh.choose_voxel((200, 40, 30)) == 0.04


def test_export_builds_the_tsdf_mesh_from_its_own_dense_pass_only_when_asked(tmp_path, monkeypatch):
    import json
    from test_nav import job, ray_cast
    from plypipe.stages import export

    def render(ply, views, antialiased=True, colour=False):
        for v in ray_cast(ply, views, antialiased):
            v.rgb = np.broadcast_to(np.array(COLOUR, np.uint8), (v.height, v.width, 3)).copy() if colour else None
            yield v

    monkeypatch.setattr(dense, "render_views", render)
    (tmp_path / "off").mkdir()
    ctx = job(tmp_path / "off", floor_points=200)
    assert "tsdfMesh" not in export._dense_occupancy(ctx, tmp_path / "off" / "scene_clean.ply", 30, True) and not (tmp_path / "off" / "mesh.glb").exists()
    monkeypatch.setattr(mesh, "choose_voxel", lambda extent, **kw: 0.04)            # a coarse voxel keeps this test short
    (tmp_path / "on").mkdir()
    ctx = job(tmp_path / "on", floor_points=200)
    ctx.opts["mesh"] = "tsdf"
    rep = export._dense_occupancy(ctx, tmp_path / "on" / "scene_clean.ply", 30, True)
    got = rep["tsdfMesh"]
    assert got["built"] and got["voxelCm"] == 4.0 and got["triangles"] <= mesh.MESH_FACES * 1.05 and rep["used"] == "depth"
    assert trimesh.load(tmp_path / "on" / "mesh.glb", force="mesh").vertices.shape[0] > 1000 and (tmp_path / "on" / "collision.glb").exists()
    scene = json.loads((tmp_path / "on" / "scene.json").read_text())
    assert scene["mesh"]["file"] == "mesh.glb" and scene["mesh"]["source"] == "tsdf" and scene["mesh"]["collision"] == "collision.glb"


def _grid(o, a, b, na, nb, flip=False):
    i, j = np.meshgrid(np.arange(na + 1), np.arange(nb + 1), indexing="ij")
    v = o + (i.reshape(-1, 1) / na) * a + (j.reshape(-1, 1) / nb) * b
    k = (np.arange(na)[:, None] * (nb + 1) + np.arange(nb)[None]).reshape(-1)
    f = np.concatenate([np.stack([k, k + nb + 1, k + 1], 1), np.stack([k + 1, k + nb + 1, k + nb + 2], 1)])
    return v, (f[:, ::-1] if flip else f)


def _noisy_hall(h, sigma, size=(25.0, 12.0, 8.0), pews=30):
    """A 25 x 12 x 8 m hall with pew boxes as one triangle soup at spacing `h`, every vertex moved by `sigma` as a real fusion's
    depth noise moves it. Returns float32 vertices and int64 faces."""
    X, Y, Z = size
    n = lambda L: max(int(round(L / h)), 1)
    e = np.eye(3)
    parts = [_grid(np.zeros(3), e[0] * X, e[2] * Z, n(X), n(Z)), _grid(e[1] * Y, e[0] * X, e[2] * Z, n(X), n(Z), True),
             _grid(np.zeros(3), e[0] * X, e[1] * Y, n(X), n(Y)), _grid(e[2] * Z, e[0] * X, e[1] * Y, n(X), n(Y), True),
             _grid(np.zeros(3), e[1] * Y, e[2] * Z, n(Y), n(Z), True), _grid(e[0] * X, e[1] * Y, e[2] * Z, n(Y), n(Z))]
    rng = np.random.default_rng(1)
    d = np.array([3.0, 0.5, 1.0])
    for _ in range(pews):
        c = np.array([rng.uniform(2, X - 4), rng.uniform(1, Y - 2), 0.0])
        for ax in range(3):
            u, w = [q for q in range(3) if q != ax]
            for s in (0, 1):
                parts.append(_grid(c + e[ax] * d[ax] * s, e[u] * d[u], e[w] * d[w], n(d[u]), n(d[w]), bool(s)))
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    v = np.concatenate(vs) + rng.normal(0, sigma, (off, 3))
    return v.astype(np.float32), np.concatenate(fs).astype(np.int64)


def test_a_noisy_hall_decimates_to_its_target_not_to_a_few_triangles():
    """The chapel's 13.9 M-triangle fusion came out at 176 triangles. The cause was not the size of the scene: the search counted
    faces whose corners fall in three different clusters, and where noise scatters the vertices' dominant normal over all six
    sides a flat patch counts as many such faces whatever the cell, about a fifth of all faces on a surface noised by a third
    of a voxel; once that floor was above the target no cell size reached it and the search ended at the coarsest, where the
    duplicates collapse to a handful. The floor scales with the face count, so 0.9 M faces against targets of 25 k and 1.7 k
    (the chapel's 13.9 M against 300 k and 20 k is the same 36:1 and 600:1) expose it in seconds."""
    v, f = _noisy_hall(0.06, 0.02)
    assert 800_000 < len(f) < 1_100_000
    for target in (25_000, 1_700):
        ov, of, _ = mesh.decimate(v, f, None, target)
        assert 0.8 * target <= len(of) <= 1.05 * target, (target, len(of))
        assert (ov.max(0) - ov.min(0)).min() > 7.0 and ov.max(0)[0] > 24.0


def _plates(count, side, gap=1.5):
    """`count` square plates of `side` metres (two triangles per 5 cm cell), each apart from the others, as one mesh."""
    vs, fs, off = [], [], 0
    for i in range(count):
        v, f = _grid(np.array([gap * (i % 20), 0.0, gap * (i // 20)]), np.array([side, 0, 0.0]), np.array([0, 0, side]), int(side / 0.05), int(side / 0.05))
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int64)


def test_connected_pieces_match_scipy_and_the_speck_rule_is_a_physical_area():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    v, f = _plates(60, 0.4)
    lab = mesh.components(len(v), f)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]]])
    n, ref = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v))), directed=False)
    assert len(np.unique(lab[f[:, 0]])) == 60 == len(np.unique(ref[f[:, 0]])) and (np.unique(np.c_[lab[f[:, 0]], ref[f[:, 0]]], axis=0).shape[0] == 60)
    pv, pf = _plates(10, 0.4)
    bv, bf = _plates(10, 1.0)
    allv, allf = np.concatenate([pv, bv + [0, 5, 0]]).astype(np.float32), np.concatenate([pf, bf + len(pv)])
    kv, kf, _, st = mesh.drop_specks(allv, allf, None)
    assert st["droppedPieces"] == 10 and abs(st["droppedM2"] - 1.6) < 0.01 and abs(st["keptM2"] - 10.0) < 0.01 and len(kf) == len(bf)
    v2, f2 = _plates(1000, 0.55)   # a fragmented fusion: 1000 pieces of 0.30 m2, each 0.1 % of the triangles: the share rule took all of it
    assert len(mesh.drop_specks(v2, f2, None)[1]) == len(f2)


def test_a_mesh_that_is_a_tenth_of_its_target_or_a_fraction_of_the_rooms_is_not_written(fused, built, tmp_path, monkeypatch):
    import json
    from test_nav import job
    from plypipe.stages import export

    real = mesh.decimate
    tiny = lambda v, f, c, target, **kw: (v[:13], np.array([[i, (i + 1) % 13, (i + 2) % 13] for i in range(13)]), None if c is None else c[:13])
    monkeypatch.setattr(mesh, "decimate", tiny)
    ctx = job(tmp_path, floor_points=200)
    rep = export._tsdf_mesh(ctx, fused)
    assert rep["built"] is False and "under 10%" in rep["why"] and rep["triangles"] == 13 and rep["rawTriangles"] > 1000
    assert not (tmp_path / "mesh.glb").exists() and not (tmp_path / "collision.glb").exists()
    assert "mesh" not in json.loads((tmp_path / "scene.json").read_text())
    logged = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(e["event"] == "warn" and "not shipped" in e["msg"] and "under 10%" in e["msg"] for e in logged)
    monkeypatch.setattr(mesh, "decimate", real)
    lo, hi = built[2].bounds
    whole = {"min": lo - 0.2, "max": hi + 0.2, "surfaceM2": 120.0}
    rep = fused.finish(tmp_path, {"min": whole["min"], "max": whole["max"] + 4 * (hi - lo), "surfaceM2": 120.0})   # dense points over five times the mesh
    assert rep["built"] is False and "extent" in rep["why"] and not (tmp_path / "mesh.glb").exists()
    rep = fused.finish(tmp_path, whole)
    assert rep["built"] is True and rep["denseSurfaceM2"] == 120.0 and (tmp_path / "mesh.glb").exists() and rep["specks"]["keptM2"] > 50
