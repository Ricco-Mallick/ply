"""The dense geometry pass (plypipe.dense) on depth that is ray-cast exactly from a box room, so no splat and no GPU:
what a perfect twin would render. The core runs on numpy here and in CI; where torch is installed one test runs it on
tensors too and compares, because that is how export runs it."""
import tracemalloc

import numpy as np
import pytest

from plypipe import dense

from synth import ROOM, camera_loop, depth_views, look, solid_room, surface_distance, write_synthetic_model

BOXES = solid_room()
WALLS = [(0, 0.0, (1, 0, 0)), (0, ROOM[0], (-1, 0, 0)), (2, 0.0, (0, 0, 1)), (2, ROOM[1], (0, 0, -1))]  # axis, where, inward normal


def cameras():
    """Two laps of the room: one looking at the walls, one looking across it."""
    (c1, r1), (c2, r2) = camera_loop(18), camera_loop(18, inward=True)
    return np.concatenate([c1, c2]), np.concatenate([r1, r2])


def cloud(views, **kw):
    stats = {}
    out = list(dense.consistent_points(views, stats=stats, **kw))
    return np.concatenate([p.points for p in out]).astype(float), np.concatenate([p.normals for p in out]).astype(float), stats, out


@pytest.fixture(scope="module")
def room():
    views = depth_views(BOXES, *cameras())
    return views, *cloud(views)


def test_wall_points_land_within_two_centimetres(room):
    _, pts, _, stats, _ = room
    err = surface_distance(BOXES, pts)
    assert stats["kept"] > 0.9 * stats["usable"] > 0.85 * stats["pixels"]      # the room survives the check
    for axis, at, _ in WALLS:
        assert (np.abs(pts[:, axis] - at) < 0.02).sum() > 100_000
    assert err.max() < 1e-4                                                     # exact depth comes back exactly


def test_noisy_depth_still_lands_on_the_walls(room):
    """0.2 % of noise on every pixel, 6 mm at 3 m: the 2 % check passes it and adds nothing to it."""
    rng = np.random.default_rng(0)
    views = [dense.View(v.K, v.R, v.t, v.width, v.height, v.name, (v.depth * (1 + rng.normal(0, 0.002, v.depth.shape))).astype(np.float32), v.alpha)
             for v in room[0]]
    pts, _, stats, _ = cloud(views)
    err = surface_distance(BOXES, pts)
    assert stats["kept"] > 0.95 * room[3]["kept"]
    assert np.percentile(err, 99) < 0.02 and err.max() < 0.05


def test_numpy_and_torch_keep_the_same_points(room):
    torch = pytest.importorskip("torch")
    views, pts, normals, stats, _ = room
    tensors = [dense.View(v.K, v.R, v.t, v.width, v.height, v.name, torch.from_numpy(v.depth), torch.from_numpy(v.alpha)) for v in views]
    t_pts, t_normals, t_stats, out = cloud(tensors)
    assert isinstance(out[0].points, np.ndarray) and out[0].points.dtype == np.float32
    assert abs(stats["kept"] - t_stats["kept"]) <= 1e-4 * stats["kept"]        # a pixel on the 2 % line may round either way
    if stats["kept"] == t_stats["kept"]:
        assert np.abs(pts - t_pts).max() < 1e-4 and np.abs(normals - t_normals).max() < 1e-3


def test_floaters_seen_from_one_place_are_removed(room):
    """A translucent blob 1 m in front of every sixth camera, drawn in that view only: its expected depth is a point
    in mid-air between the blob and the wall, and every other view sees the wall there."""
    c, r = cameras()
    views = depth_views(BOXES, c, r, blobs=[(c[i] + 1.0 * r[i][2], 0.12, 0.6, i) for i in range(0, len(c), 6)])
    before = surface_distance(BOXES, cloud(views, min_agree=0)[0])
    pts, _, stats, _ = cloud(views)
    assert (before > 0.05).sum() > 5_000                                        # the floaters are in the depth
    assert (surface_distance(BOXES, pts) > 0.02).sum() == 0                     # and in none of the points kept
    assert stats["kept"] > 0.98 * room[3]["kept"]                               # at the cost of the pixels they covered


def test_a_translucent_ball_in_the_splat_leaves_no_points_behind_it(room):
    """A ball that every view renders, in the middle of the room: each view puts a point in mid-air between the ball
    and the wall behind, and the views to either side, which see past the ball, do not find it there."""
    views = depth_views(BOXES, *cameras(), blobs=[((3.0, 1.2, 2.0), 0.15, 0.5, None)])
    before = surface_distance(BOXES, cloud(views, min_agree=0)[0])
    pts, _, _, _ = cloud(views)
    assert (before > 0.05).sum() > 10_000 and (surface_distance(BOXES, pts) > 0.05).sum() == 0


def test_a_surface_that_more_views_look_through_than_confirm_is_dropped():
    """Three views in a row of seven draw a panel in mid-air and agree on it; the four beside them see the wall 1.5 m
    behind where it should be. Two agreeing views would be enough: the views that look through it are what removes it."""
    centers = np.c_[np.linspace(1.9, 3.1, 7), np.full(7, 1.45), np.full(7, 1.0)]
    rots = np.array([look(np.pi / 2, 0.0)] * 7)
    views = depth_views(BOXES, centers, rots)
    views[2:5] = depth_views(np.vstack([BOXES, [2.3, 1.0, 2.5, 2.7, 1.8, 2.52]]), centers, rots)[2:5]
    on_panel = lambda pts: int((np.abs(pts[:, 2] - 2.5) < 0.01).sum())
    pts, _, stats, _ = cloud(views)
    trusting, _, _, _ = cloud(views, through=np.inf)
    assert on_panel(trusting) > 1_000 and on_panel(pts) == 0
    assert stats["contradicted"] >= on_panel(trusting)


def test_normals_on_the_walls_point_into_the_room(room):
    _, pts, normals, _, out = room
    W, D, H = ROOM
    assert np.abs(np.linalg.norm(normals, axis=1) - 1).max() < 1e-3
    for p in out:                                                               # every normal faces the camera that saw it
        assert ((p.center - p.points) * p.normals).sum(1).min() > 0
    for axis, at, inward in WALLS:
        on = np.abs(pts[:, axis] - at) < 1e-3
        along = pts[on][:, 2 - axis]
        crease = np.minimum.reduce([pts[on][:, 1], H - pts[on][:, 1], along, (D if axis == 0 else W) - along])
        angle = np.degrees(np.arccos(np.clip(normals[on] @ np.array(inward, float), -1, 1)))
        assert (crease > 0.15).sum() > 100_000 and angle[crease > 0.15].max() < 0.5
        # the rest are pixels whose neighbours lie on the floor or the next wall: their normal is a blend of the two
        assert (angle > 3).mean() < 0.02


def test_neighbours_are_chosen_by_where_a_view_stands_and_looks():
    """Down a corridor and back: at the turn, the next frame in time looks the other way and is no neighbour; and a
    second view from the same spot is left out however alike it looks."""
    x = np.arange(0, 3.01, 0.5)
    out = np.c_[x, np.full(len(x), 1.45), np.zeros(len(x))]
    centers = np.concatenate([out, out[::-1] + [0, 0, 0.3], out[3:4]])
    forwards = np.array([[1.0, 0, 0]] * len(x) + [[-1.0, 0, 0]] * len(x) + [[1.0, 0, 0]])
    near = dense.neighbours(centers, forwards, np.full(len(centers), 3.0), k=4)
    turn, same_spot = len(x) - 1, len(centers) - 1
    assert set(near[turn]) <= set(range(len(x))) | {same_spot}                  # all four look the way it does
    assert same_spot not in near[3] and 3 not in near[same_spot]                # no baseline: agreement would mean nothing
    assert set(near[3]) == {1, 2, 4, 5}
    alone = dense.neighbours(centers[:2], forwards[:2], np.full(2, 3.0), k=4)
    assert alone.tolist() == [[1, -1, -1, -1], [0, -1, -1, -1]]


def test_points_are_streamed_view_by_view(room):
    """Memory is the usable depth of every view and one view's points, whatever the number of views."""
    views = room[0] * 2
    held = sum(v.depth.nbytes for v in views)
    stream = dense.consistent_points(iter(views))
    tracemalloc.start()
    kept, saved = 0, dense.Cloud(voxel=0.05)
    for p in stream:
        kept += len(p.points)
        saved.add(p.points, p.normals)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    everything = kept * 24                                                      # float32 points and normals of every view
    assert everything > 4 * held
    assert peak < held + 0.15 * everything


def test_the_saved_cloud_has_one_point_per_voxel_and_a_cap(tmp_path):
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (50_000, 3)).astype(np.float32)
    nrm = np.tile(np.array([[0, 1, 0]], np.float32), (len(pts), 1))
    saved = dense.Cloud(voxel=0.1)
    for part in np.split(np.arange(len(pts)), 10):
        saved.add(pts[part], nrm[part])
    kept, _ = saved.arrays()
    cells = np.floor(kept / 0.1).astype(int)
    assert len(saved) == len(kept) == len(np.unique(cells, axis=0)) == 1000     # every 10 cm cube once, across the batches
    assert (kept[(cells == np.floor(pts[0] / 0.1).astype(int)).all(1)] == pts[0]).all()  # the first point seen in a cube is the one kept
    saved.add(pts, nrm)
    assert len(saved) == 1000                                                   # seen before: nothing new
    capped = dense.Cloud(voxel=0.1, cap=300)
    capped.add(pts, nrm)
    assert capped.voxel == pytest.approx(0.2) and len(capped) == 125            # over the cap: twice the voxel
    rep = capped.save(tmp_path / dense.FILE)
    back = np.load(tmp_path / dense.FILE)
    assert rep["points"] == 125 and back["points"].shape == (125, 3) and back["points"].dtype == np.float32
    assert back["normals"].dtype == np.float16 and float(back["voxel"]) == pytest.approx(0.2)


def test_training_views_are_every_fifth_at_half_size(tmp_path):
    write_synthetic_model(tmp_path / "sparse" / "0")                            # 60 photos, PINHOLE 1600x1200, f 1200
    views = dense.training_views(tmp_path, every=30, step=5)
    train = [f"{i:06d}.jpg" for i in range(60) if i % 30]
    assert [v.name for v in views] == train[::5]                                # the held-out views are not rendered
    v = views[0]
    assert (v.width, v.height) == (800, 600) and v.K[0].tolist() == [600, 0, 400] and v.K[1].tolist() == [0, 600, 300]
    assert np.allclose(v.R @ v.center + v.t, 0) and np.allclose(v.R @ v.R.T, np.eye(3))
    cams = tmp_path / "sparse" / "0" / "cameras.txt"
    cams.write_text(cams.read_text().replace("PINHOLE 1600 1200 1200 1200 800 600", "SIMPLE_RADIAL 1600 1200 1200 800 600 0.01"))
    assert dense.training_views(tmp_path) is None


def test_rendering_needs_a_gpu(tmp_path):
    assert dense.render_views(tmp_path / "missing.ply", []) is None


def test_the_renderer_asks_gsplat_for_expected_depth(tmp_path):
    """The GPU loop on a CPU, with a stand-in for gsplat's rasteriser that checks what it is handed."""
    torch = pytest.importorskip("torch")
    props = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros((7, len(props)), np.float32)
    data[:, :3], data[:, 3], data[:, 4:7], data[:, 7] = np.arange(21).reshape(7, 3), 2.0, np.log(0.05), 1.0
    ply = tmp_path / "scene_clean.ply"
    ply.write_bytes(("ply\nformat binary_little_endian 1.0\nelement vertex 7\n" + "".join(f"property float {p}\n" for p in props)
                     + "end_header\n").encode() + data.tobytes())
    views = depth_views(BOXES, *camera_loop(3))
    calls = []

    def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kw):
        assert means.shape == (7, 3) and quats.shape == (7, 4) and scales.shape == (7, 3) and opacities.shape == (7,)
        assert torch.allclose(scales, torch.full((7, 3), 0.05)) and torch.allclose(opacities, torch.sigmoid(torch.tensor(2.0)))
        assert viewmats.shape == (1, 4, 4) and Ks.shape == (1, 3, 3) and viewmats.dtype == Ks.dtype == torch.float32
        assert colors.shape[0] == 7 and not torch.is_grad_enabled()
        calls.append((kw, viewmats[0].numpy(), Ks[0].numpy()))
        return torch.full((1, height, width, 1), 2.5), torch.ones(1, height, width, 1), {}

    out = list(dense._render(ply, views, True, "cpu", rasterization))
    assert [c[0] for c in calls] == [{"render_mode": "ED", "rasterize_mode": "antialiased"}] * 3
    for v, o, (_, viewmat, K) in zip(views, out, calls):
        assert np.allclose(viewmat[:3, :3], v.R, atol=1e-6) and np.allclose(viewmat[:3, 3], v.t, atol=1e-6) and np.allclose(K, v.K)
        assert o.depth.shape == o.alpha.shape == (v.height, v.width) and float(o.depth[0, 0]) == 2.5 and o.name == v.name
    assert sum(len(p.points) for p in dense.consistent_points(out)) == 0        # a flat 2.5 m everywhere is no room: nothing agrees
