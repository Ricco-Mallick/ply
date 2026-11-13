"""plypipe.mushroom: the dataset's pose files, the fit of a twin's cameras onto them, and the laser reference in the twin's frame.
Every fixture is written here in the dataset's own file formats (checked against the honka and koivu archives, 2026-11-13)."""
import json
import struct
import sys
import types

import numpy as np
import pytest

from plypipe import geometry, mushroom

ROOM = np.array([6.0, 4.0, 2.5])  # laser frame: z up, one corner at the origin


def look_at(pos, target):
    """Camera-to-world with OpenCV axes (x right, y down, z forward) in a z-up world."""
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    M = np.eye(4)
    M[:3, :3] = np.stack([r, np.cross(f, r), f], axis=1)
    M[:3, 3] = pos
    return M


def walk(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([3 + 2.0 * np.cos(ang), 2 + 1.2 * np.sin(ang), 1.4 + 0.15 * rng.standard_normal(n)], axis=1)
    return np.array([look_at(p, [3 + 0.3 * rng.standard_normal(), 2 + 0.3 * rng.standard_normal(), 1.2]) for p in pos])


def dataset_json(c2w_cv, per_frame_intrinsics=False, gaps=True):
    """transformations.json as the iPhone folder has it: OpenGL poses, frame numbers from 1 with gaps."""
    gl = c2w_cv @ mushroom.GL_TO_CV
    K = {"fl_x": 807.9, "fl_y": 807.9, "cx": 366.8, "cy": 503.7, "h": 994, "w": 738}
    nums = [i + 1 + (i // 7 if gaps else 0) for i in range(len(gl))]
    frames = [{**(K if per_frame_intrinsics else {}), "file_path": f"./images/frame_{k:05d}.jpg", "depth_file_path": f"./depth/frame_{k:05d}.png",
               "transform_matrix": m.tolist()} for k, m in zip(nums, gl)]
    return {**({} if per_frame_intrinsics else K), "camera_model": "OPENCV", "frames": frames}


def random_similarity(rng, scale):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    R = q * np.sign(np.linalg.det(q))
    return scale, R, rng.uniform(-3, 3, 3)


@pytest.mark.parametrize("per_frame", [False, True])
def test_pose_file_round_trips_a_synthetic_walk_in_the_datasets_format(tmp_path, per_frame):
    c2w = walk()
    path = tmp_path / "transformations.json"
    path.write_text(json.dumps(dataset_json(c2w, per_frame_intrinsics=per_frame)))
    poses = mushroom.read_poses(path)
    assert poses.names[:3] == ["frame_00001", "frame_00002", "frame_00003"] and len(poses.names) == 60
    assert np.abs(poses.c2w - c2w).max() < 1e-12
    assert poses.intrinsics == {"fx": 807.9, "fy": 807.9, "cx": 366.8, "cy": 503.7, "w": 738.0, "h": 994.0}


def test_opengl_poses_look_at_the_wall_the_camera_faces():
    cam = look_at(np.array([3.0, 2.0, 1.4]), [6.0, 2.0, 1.4])  # faces the x = 6 wall
    meta = dataset_json(cam[None])
    forward_gl = -np.array(meta["frames"][0]["transform_matrix"])[:3, 2]  # an OpenGL camera looks down -z
    assert forward_gl @ [1, 0, 0] > 0.999
    assert mushroom.read_poses(meta).c2w[0][:3, 2] @ [1, 0, 0] > 0.999  # and the OpenCV matrix down +z


def test_alignment_file_holds_the_matrix_flat_or_square():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    assert np.allclose(mushroom.read_alignment({"gt_transformation": T.ravel().tolist()}), T)
    assert np.allclose(mushroom.read_alignment({"gt_transformation": T.tolist()}), T)


def test_alignment_direction_is_the_one_that_puts_the_cameras_inside_the_scan():
    rng = np.random.default_rng(1)
    scan = np.concatenate([rng.uniform(0, 1, (4000, 3)) * ROOM, np.stack([np.zeros(500), rng.uniform(0, 4, 500), rng.uniform(0, 2.5, 500)], 1)])
    centres = walk()[:, :3, 3]
    s, R, t = random_similarity(rng, 1.0)
    pose_to_scan = np.eye(4)
    pose_to_scan[:3, :3], pose_to_scan[:3, 3] = R, t
    pose_frame = geometry.apply(np.linalg.inv(pose_to_scan), centres)
    for stored in (pose_to_scan, np.linalg.inv(pose_to_scan)):
        got, info = mushroom.orient_alignment(stored, pose_frame, scan)
        assert np.allclose(geometry.apply(got, pose_frame), centres, atol=1e-9)
        assert info["direction"] == ("as-stored" if stored is pose_to_scan else "inverted") and info["confident"]


def test_scan_frame_poses_refuse_a_scaling_alignment():
    poses = mushroom.read_poses(dataset_json(walk(5)))
    scaled = np.diag([2.0, 2.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="rigid"):
        mushroom.scan_frame_poses(poses, scaled)
    moved = np.eye(4)
    moved[:3, 3] = [1, 0, 0]
    assert np.allclose(mushroom.scan_frame_poses(poses, moved).c2w[:, 0, 3], poses.c2w[:, 0, 3] + 1)


def test_similarity_recovers_scale_rotation_and_translation_with_a_fifth_of_the_poses_wrong():
    rng = np.random.default_rng(2)
    ref = walk(100)[:, :3, 3]
    s, R, t = random_similarity(rng, 0.37)
    twin = ((ref - t) @ R) / s + rng.normal(0, 0.002, ref.shape)  # twin units: ref = s R twin + t
    bad = rng.choice(len(ref), 20, replace=False)
    twin[bad] += rng.uniform(1.0, 4.0, (20, 3)) * rng.choice([-1, 1], (20, 3))
    sim = mushroom.fit_similarity(twin, ref)
    assert abs(sim.scale / s - 1) < 1e-3
    assert np.abs(sim.R - R).max() < 1e-3 and np.abs(sim.t - t).max() < 5e-3
    assert not sim.inliers[bad].any() and sim.rms < 0.01 and sim.n == 100


def test_scale_error_is_signed_percent_of_how_much_longer_the_twins_metres_are():
    rng = np.random.default_rng(3)
    ref = walk(80)[:, :3, 3]
    _, R, t = random_similarity(rng, 1)
    for stretch, want in ((1.08, 8.0), (0.95, -5.0), (1.0, 0.0)):
        twin = (ref - t) @ R * stretch  # a twin whose lengths are `stretch` times the room's
        assert mushroom.scale_error_pct(mushroom.fit_similarity(twin, ref)) == pytest.approx(want, abs=0.05)


def test_twin_cameras_match_the_reference_by_renumbered_index_or_by_name():
    ref = [f"frame_{k:05d}" for k in (1, 2, 3, 5, 6, 8)]
    rt, rr = mushroom.match(["000000.jpg", "000003.jpg", "000005.jpg", "000099.jpg"], ref)
    assert rt.tolist() == [0, 1, 2] and rr.tolist() == [0, 3, 5]
    rt, rr = mushroom.match(["frame_00005.jpg", "frame_00004.jpg"], ref)
    assert rt.tolist() == [0] and rr.tolist() == [3]


def test_reference_in_the_twins_frame_lands_on_the_known_wall():
    rng = np.random.default_rng(4)
    ref_c = walk(80)[:, :3, 3]
    s, R, t = random_similarity(rng, 0.9)
    twin_c = ((ref_c - t) @ R) / s
    names = [f"{i:06d}.jpg" for i in range(80)]
    sim = mushroom.match_and_fit(names, twin_c, mushroom.input_order([f"frame_{i + 1:05d}" for i in range(80)]), ref_c)
    wall = np.stack([np.full(500, 6.0), rng.uniform(0, 4, 500), rng.uniform(0, 2.5, 500)], axis=1)  # laser frame: the x = 6 wall
    in_twin = mushroom.reference_to_twin(sim, wall)
    normal, offset = R[0], (6.0 - t[0]) / s  # that wall in twin units: n . p = d, from scan x = s * R[0] . p + t[0]
    assert np.abs(in_twin @ normal - offset).max() < 1e-3 * abs(offset)
    assert np.allclose(mushroom.twin_to_reference(sim, in_twin), wall, atol=1e-6)


def write_ply(path, verts, faces=None, big=False):
    e = ">" if big else "<"
    head = ["ply", f"format binary_{'big' if big else 'little'}_endian 1.0", f"element vertex {len(verts)}", "property float x", "property float y",
            "property float z", "property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        head += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    body = b"".join(struct.pack(e + "fffBBB", *v, 1, 2, 3) for v in verts) + b"".join(struct.pack(e + "Biii", 3, *f) for f in faces or [])
    path.write_bytes(("\n".join(head + ["end_header"]) + "\n").encode() + body)


@pytest.mark.parametrize("big", [False, True])
def test_ply_points_read_from_a_mesh_with_faces_and_a_stride(tmp_path, big):
    verts = np.random.default_rng(5).uniform(-5, 5, (50, 3)).astype(np.float32)
    write_ply(tmp_path / "gt_mesh.ply", verts, faces=[(0, 1, 2), (3, 4, 5)], big=big)
    assert np.array_equal(mushroom.read_ply_points(tmp_path / "gt_mesh.ply"), verts)
    assert np.array_equal(mushroom.read_ply_points(tmp_path / "gt_mesh.ply", stride=5), verts[::5])


def test_thin_keeps_one_point_per_cell_and_respects_the_limit():
    pts = np.random.default_rng(6).uniform(0, 1, (5000, 3))
    out = mushroom.thin(pts, 0.1)
    assert 500 < len(out) <= 1000 and len({tuple(np.floor(p / 0.1).astype(int)) for p in out}) == len(out)
    assert len(mushroom.thin(pts, 0.1, limit=100)) == 100


@pytest.mark.parametrize("scipy_present", [True, False])
def test_cloud_scores_count_a_shifted_cloud_by_its_distance(monkeypatch, scipy_present):
    if not scipy_present:
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)  # the worker image has no scipy: the torch path must give the same numbers
    ref = np.stack(np.meshgrid(*[np.arange(0, 3, 0.5)] * 3), -1).reshape(-1, 3)  # a lattice: the nearest point is the shifted twin itself
    for shift, f5, f10 in ((0.03, 1.0, 1.0), (0.07, 0.0, 1.0), (0.2, 0.0, 0.0)):
        got = geometry.cloud_scores(ref + [shift, 0, 0], ref)
        assert got["f5cm"]["f"] == pytest.approx(f5, abs=0.02) and got["f10cm"]["f"] == pytest.approx(f10, abs=0.02)
    assert geometry.cloud_scores(ref, ref)["chamferM"] == 0.0


def write_text_model(folder, names, c2w_cv):
    folder.mkdir(parents=True)
    (folder / "cameras.txt").write_text("1 SIMPLE_RADIAL 738 994 800 369 497 0\n")
    lines = []
    for i, (name, M) in enumerate(zip(names, c2w_cv)):
        w2c = np.linalg.inv(M)
        q = geometry.mat_to_quat(w2c[:3, :3])
        lines += [f"{i + 1} {' '.join(f'{v:.12g}' for v in q)} {' '.join(f'{v:.12g}' for v in w2c[:3, 3])} 1 {name}", ""]
    (folder / "images.txt").write_text("\n".join(lines) + "\n")
    (folder / "points3D.txt").write_text("1 0 0 0 1 2 3 0.5 1 0\n")


def test_score_job_reports_the_scale_error_and_the_splats_against_the_scan(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    c2w = walk(60)
    ref_c = c2w[:, :3, 3]
    scan = rng.uniform(0, 1, (6000, 3)) * ROOM
    s, R, t = random_similarity(rng, 1 / 1.06)  # a twin 6% too large
    inv = np.eye(4)
    inv[:3, :3], inv[:3, 3] = R.T / s, -(R.T @ t) / s  # scan -> twin
    twin_c2w = c2w.copy()  # only the centres matter to the fit, so the rotations stay the walk's
    twin_c2w[:, :3, 3] = geometry.apply(inv, ref_c)
    names = [f"{i:06d}.jpg" for i in range(60)]
    write_text_model(tmp_path / "job" / "data" / "sparse" / "0", names, twin_c2w)
    (tmp_path / "job" / "scene.ply").write_bytes(b"")
    twin_pts = geometry.apply(inv, scan[:2000])
    splats = types.SimpleNamespace(x=twin_pts[:, 0], y=twin_pts[:, 1], z=twin_pts[:, 2], alpha=np.ones(2000))
    monkeypatch.setattr("plypipe.topview.read_ply", lambda p: splats)
    reference = {"meta": {"direction": {"direction": "as-stored"}}, "names": mushroom.input_order([f"frame_{i + 1:05d}" for i in range(60)]),
                 "centres": ref_c, "points": scan}
    out = mushroom.score_job(tmp_path / "job", reference)
    assert out["scaleErrorPct"] == pytest.approx(6.0, abs=0.1) and out["cameras"]["matched"] == 60
    assert out["splats"]["f5cm"]["precision"] > 0.99 and 0.2 < out["splats"]["f5cm"]["recall"] < 0.5  # 2,000 of 6,000 scan points seen


def test_reference_folder_loads_what_the_seed_writes(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"names": ["frame_00001"], "centres": [[1, 2, 3]], "direction": {"direction": "inverted"}}))
    np.save(tmp_path / "ref_points.npy", np.zeros((4, 3), np.float32))
    ref = mushroom.load_reference(tmp_path)
    assert ref["names"] == ["frame_00001"] and ref["centres"].shape == (1, 3) and ref["points"].shape == (4, 3)
