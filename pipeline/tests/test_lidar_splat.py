"""Route 8a: a laser-scanned room becomes a twin. The scan is lidar_scene.py's: synth.py's room and table and sofa as solid
boxes, seen from two stations, so the far sides of the boxes and the floor behind them are shadows, with 2 mm of noise."""

import json
import os
import shutil

import cv2
import numpy as np
import pytest

from plypipe import clean, geometry, lidar_splat as L, walkcheck
from plypipe.cli import main
from lidar_scene import STATIONS, scanned_room, to_file_frame

TRAINER_LAYOUT = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + \
                 ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
FLOOR, CEILING, WALLS = 0, 1, (2, 3, 4, 5)


@pytest.fixture(scope="module")
def scan():
    return scanned_room()


def write_npz(path, f: dict) -> None:
    origin = f["points"].mean(0)
    np.savez(path, points=(f["points"] - origin).astype(np.float32), colors=f["colors"], stations=(f["stations"] - origin).astype(np.float32), origin=origin)


@pytest.fixture(scope="module")
def twin(tmp_path_factory, scan):
    """plypipe lidar on the scan as a z-up file far from the origin, up to the splat file (the export is tested apart)."""
    root = tmp_path_factory.mktemp("lidar")
    f = to_file_frame(scan)
    write_npz(root / "scan.npz", f)
    out = root / "job"
    assert main(["lidar", str(root / "scan.npz"), "--out", str(out), "--no-export", "--quiet"]) == 0
    scene = json.loads((out / "scene.json").read_text())
    _, c = clean.read_ply(out / "scene.ply")
    T = np.array(scene["transform"])
    d = L.load_points(root / "scan.npz")
    # the scan in the scene frame, as the stage computes it, with the true normals turned the same way
    q = (d["points"].astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    return {"out": out, "scene": scene, "cols": c, "T": T, "report": json.loads((out / "report.json").read_text())["lidar"],
            "scan_scene": q, "colors": d["colors"], "normals_scene": f["normals"] @ T[:3, :3].T}


def centres(c) -> np.ndarray:
    return np.stack([c["x"], c["y"], c["z"]], 1).astype(np.float64)


def axes_of(c) -> np.ndarray:
    """The disks' rotation matrices [N, 3, 3] (columns are their axes) from the file's (w, x, y, z) quaternions."""
    w, x, y, z = (c[f"rot_{i}"].astype(np.float64) for i in range(4))
    R = np.empty((len(w), 3, 3))
    R[:, 0] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], 1)
    R[:, 1] = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], 1)
    R[:, 2] = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], 1)
    return R.transpose(0, 2, 1)


def test_the_rotation_helper_agrees_with_geometry_for_every_branch():
    rng = np.random.default_rng(1)
    qs = rng.normal(size=(500, 4))
    qs = np.vstack([qs, [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0.7071, 0.7071, 0]]])
    R = np.array([geometry.quat_to_mat(q) for q in qs])
    back = L._quat(R)
    assert np.allclose(np.array([geometry.quat_to_mat(q) for q in back]), R, atol=1e-9)


def test_splat_centres_hold_the_scan_to_a_centimetre(twin):
    f = L.fscore(centres(twin["cols"]), twin["scan_scene"], 0.01)
    assert f["f"] >= 0.95 and f["precision"] >= 0.95 and f["recall"] >= 0.95, f
    assert twin["report"]["fScore"]["f"] >= 0.95 and twin["report"]["voxelM"] == pytest.approx(0.01)


def test_every_disk_faces_the_surface_it_sits_on_and_the_station_that_saw_it(twin, scan):
    from scipy.spatial import cKDTree
    c = twin["cols"]
    n = axes_of(c)[:, :, 2]
    assert np.allclose(n, np.stack([c["nx"], c["ny"], c["nz"]], 1), atol=1e-3)
    p = centres(c)
    idx = cKDTree(twin["scan_scene"]).query(p)[1]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n, twin["normals_scene"][idx]), -1, 1)))
    flat = ~scan["edge"][idx]
    assert flat.mean() > 0.7
    assert (ang[flat] > 10.0).mean() < 2e-4, np.percentile(ang[flat], [50, 99, 99.99, 100])  # a handful of 270k, at corners the edge test missed
    assert np.percentile(ang[flat], 99) < 6.0
    assert (ang <= 10.0).mean() > 0.97  # the rest sit within 3 cm of an edge, where two surfaces meet


def test_a_station_in_the_next_room_does_not_turn_a_walls_normals_away():
    yy, zz = np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 4, 0.02))
    face = lambda x: np.stack([np.full(yy.size, x), yy.ravel(), zz.ravel()], 1).astype(np.float32)
    pts = np.concatenate([face(0.0), face(-0.2)])  # a 20 cm wall scanned from both sides
    stations = np.array([[2.0, 1.0, 2.0], [-2.0, 1.0, 2.0], [-1.0, 1.0, 2.0]])  # the third is nearest to the room-side face, behind it
    rng = np.random.default_rng(0)
    normals = np.tile([1.0, 0, 0], (len(pts), 1)).astype(np.float32) * rng.choice([-1.0, 1.0], (len(pts), 1)).astype(np.float32)
    out = L.orient(pts, normals, stations)
    half = yy.size
    assert (out[:half, 0] > 0).all() and (out[half:, 0] < 0).all()


def test_no_disk_is_wider_than_two_voxels_and_each_is_thin(twin):
    c, v = twin["cols"], twin["report"]["voxelM"]
    s = np.exp(np.stack([c["scale_0"], c["scale_1"], c["scale_2"]], 1))
    assert (4 * s[:, :2].max(1) <= 2 * v * (1 + 1e-5)).all()
    assert (s[:, 2] <= 0.11 * s[:, :2].min(1)).all() and (s > 0).all()
    assert np.allclose(1 / (1 + np.exp(-c["opacity"])), 0.95, atol=1e-4)


def test_the_ply_has_the_trainers_layout_and_its_colours_round_trip(twin, tmp_path):
    names = [ln.split()[-1] for ln in (twin["out"] / "scene.ply").read_bytes()[:3000].decode("latin-1").split("end_header")[0].splitlines() if ln.startswith("property")]
    assert names == TRAINER_LAYOUT
    rng = np.random.default_rng(0)
    p = rng.random((50, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (50, 3)).astype(np.uint8)
    nrm = np.tile([0.0, 1.0, 0.0], (50, 1)).astype(np.float32)
    L.write_ply(tmp_path / "a.ply", L.disks(p, rgb, nrm, np.tile([1.0, 0, 0], (50, 1)).astype(np.float32), np.full(50, 0.01, np.float32), 0.01))
    _, c = clean.read_ply(tmp_path / "a.ply")
    back = np.clip(0.5 + 0.28209479177387814 * np.stack([c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]], 1), 0, 1) * 255
    assert np.abs(back - rgb).max() <= 1.0
    assert all((c[k] == 0).all() for k in TRAINER_LAYOUT if k.startswith("f_rest_"))
    from plypipe.stages import export
    assert export.ply_sh_degree(tmp_path / "a.ply") == 3 and export.ply_count(tmp_path / "a.ply") == 50


def test_the_scan_colours_arrive_in_the_splats(twin):
    c = twin["cols"]
    rgb = np.clip(0.5 + 0.28209479177387814 * np.stack([c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]], 1), 0, 1) * 255
    want = L.thin_to_cap(twin["scan_scene"], twin["colors"], 0.01, L.splat_cap())[1]
    assert len(want) == len(rgb) and np.abs(rgb - want).max() <= 1.0


def test_the_cap_coarsens_the_voxel_and_says_which_size_it_used(scan, tmp_path):
    f = to_file_frame(scan)
    pts, cols = f["points"].astype(np.float32), f["colors"]
    p, c, v = L.thin_to_cap(pts - pts.mean(0), cols, 0.01, 60_000)
    assert len(p) <= 60_000 and v > 0.01 and len(c) == len(p)
    p1, _, v1 = L.thin_to_cap(pts - pts.mean(0), cols, 0.01, 10_000_000)
    assert v1 == 0.01 and len(p1) > 60_000
    write_npz(tmp_path / "scan.npz", f)
    assert main(["lidar", str(tmp_path / "scan.npz"), "--out", str(tmp_path / "job"), "--cap", "60000", "--no-export", "--quiet"]) == 0
    rep = json.loads((tmp_path / "job" / "report.json").read_text())["lidar"]
    assert rep["splats"] <= 60_000 and rep["coarsened"] and rep["voxelM"] > rep["voxelRequestedM"] == 0.01
    assert L.splat_cap() == 2_000_000


@pytest.mark.parametrize("kind", [dict(z_up=True), dict(z_up=True, flip=True), dict(z_up=False)], ids=["z up", "z down", "y up"])
def test_the_floor_goes_to_y_zero_and_the_walls_onto_the_axes(scan, kind):
    f = to_file_frame(scan, **kind)
    pts = f["points"] - f["points"].mean(0)
    T, info = L.align(pts.astype(np.float32), f["colors"], f["stations"] - f["points"].mean(0))
    q = pts @ T[:3, :3].T + T[:3, 3]
    face = scan["face"]
    assert abs(np.median(q[face == FLOOR, 1])) < 0.005 and q[face == FLOOR, 1].std() < 0.005
    assert np.median(q[face == CEILING, 1]) == pytest.approx(2.8, abs=0.01)
    for w in WALLS:  # a wall is a plane x = const or z = const
        xy = q[face == w][:, [0, 2]]
        assert min(xy[:, 0].std(), xy[:, 1].std()) < 0.01
    assert info["floorTiltDeg"] < 0.1 and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-6  # scale stays the scan's own metres


def test_up_can_be_given_and_is_then_not_second_guessed(scan):
    f = to_file_frame(scan, z_up=False)
    pts = (f["points"] - f["points"].mean(0)).astype(np.float32)
    T, info = L.align(pts, f["colors"], f["stations"] - f["points"].mean(0), up="+y")
    assert info["upEvidence"]["why"] == "given: +y" and info["up"] == [0.0, 1.0, 0.0]


def test_scene_json_places_the_twin_for_the_viewer(twin):
    s = twin["scene"]
    assert s["up"] == [0, 1, 0] and s["floorY"] == 0.0 and s["units"] == "meters" and s["handedness"] == "right"
    assert s["scale"] == 1.0 and s["scaleSource"] == "lidar" and s["shell"] == "shell.json"
    assert abs(s["spawn"]["position"][1] - 1.6) < 1e-9 and len(s["cameraPath"]) > 5 and len(s["tour"]) >= 4
    assert s["occupancy"] == {"image": "occupancy.png", "meta": "occupancy.json"}
    assert all(abs(p[1] - 1.6) < 1e-9 for p in s["cameraPath"])
    mn, mx = np.array(s["bounds"]["min"]), np.array(s["bounds"]["max"])
    assert mx[0] - mn[0] == pytest.approx(4.0, abs=0.5) and mx[2] - mn[2] == pytest.approx(6.0, abs=0.5)  # the room, turned onto the axes
    sh = json.loads((twin["out"] / "shell.json").read_text())
    assert len(sh["walls"]) == 4 and sh["ceilingY"] == pytest.approx(2.8, abs=0.03) and sh["floorY"] == 0.0


def test_the_walking_grid_blocks_the_walls_and_the_boxes_and_frees_the_floor(twin, scan):
    out = twin["out"]
    meta = json.loads((out / "occupancy.json").read_text())
    grid = cv2.imread(str(out / "occupancy.png"), cv2.IMREAD_GRAYSCALE)
    assert meta["source"] == "lidar" and grid.shape == (meta["height"], meta["width"])

    def cells(q):
        col = np.floor((q[:, 0] - meta["origin"][0]) / meta["cellSize"]).astype(int)
        row = np.floor((q[:, 2] - meta["origin"][1]) / meta["cellSize"]).astype(int)
        return grid[row, col]
    q, face = twin["scan_scene"], scan["face"]
    wall = q[np.isin(face, WALLS) & (q[:, 1] > 0.3) & (q[:, 1] < 1.8)]
    assert (cells(wall) == 0).mean() > 0.97                                  # walls block
    boxes = q[(face >= 6) & (q[:, 1] > 0.3)]
    assert (cells(boxes) == 0).mean() > 0.97                                 # the table and the sofa block
    inner = q[(face == FLOOR) & (np.abs(q[:, 0] - q[:, 0].mean()) < 1.2) & (q[:, 2] < q[:, 2].mean() - 1.2)]   # floor in the clear
    assert len(inner) > 100 and (cells(inner) == 255).mean() > 0.95        # the floor is free
    ok, xs, zs = walkcheck.standable(grid, meta)
    path = np.array(twin["scene"]["cameraPath"])
    j = np.clip(np.round((path[:, 2] - zs[0]) / walkcheck.STEP).astype(int), 0, len(zs) - 1)
    i = np.clip(np.round((path[:, 0] - xs[0]) / walkcheck.STEP).astype(int), 0, len(xs) - 1)
    assert ok[j, i].all()                                                    # the walk through the stations never leaves the floor
    assert meta["walk"]["cameras"] == 1.0 and meta["walk"]["pass"] and twin["report"]["route"]["walkable"] == len(STATIONS)


def test_a_coloured_ply_without_stations_makes_a_twin_too(scan, tmp_path):
    f = to_file_frame(scan, z_up=False)
    p = (f["points"] - f["points"].mean(0)).astype(np.float32)
    arr = np.zeros(len(p), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = p[:, 0], p[:, 1], p[:, 2]
    arr["red"], arr["green"], arr["blue"] = f["colors"][:, 0], f["colors"][:, 1], f["colors"][:, 2]
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(p))
    (tmp_path / "scan.ply").write_bytes(head.encode() + arr.tobytes())
    assert main(["lidar", str(tmp_path / "scan.ply"), "--out", str(tmp_path / "job"), "--no-export", "--quiet", "--voxel", "0.02"]) == 0
    s = json.loads((tmp_path / "job" / "scene.json").read_text())
    rep = json.loads((tmp_path / "job" / "report.json").read_text())["lidar"]
    assert s["scaleSource"] == "lidar" and rep["upEvidence"]["axis"] == "y" and len(s["cameraPath"]) == 1 and s["cameraHeight"] is None
    assert rep["route"]["stations"] == 0 and rep["walk"]["pass"] is not None


needs_npx = pytest.mark.skipif(not shutil.which("npx") or os.environ.get("PLY_SKIP_NPX") == "1", reason="needs Node/npx and network for splat-transform")


@needs_npx
def test_the_whole_twin_is_exported_to_spz_with_the_scan_options(scan, tmp_path):
    write_npz(tmp_path / "scan.npz", to_file_frame(scan))
    out = tmp_path / "job"
    assert main(["lidar", str(tmp_path / "scan.npz"), "--out", str(out), "--quiet", "--sh-degree", "0"]) == 0
    for name in ("scene.spz", "scene_lite.spz", "scene.json", "occupancy.json", "occupancy.png", "shell.json"):
        assert (out / name).stat().st_size > 100, name
    import gzip
    for name in ("scene.spz", "scene_lite.spz"):
        head = gzip.open(out / name).read(16)
        assert int.from_bytes(head[:4], "little") == 0x5053474E and int.from_bytes(head[4:8], "little") == 3
    rep = json.loads((out / "report.json").read_text())
    ex, lid = rep["export"], rep["lidar"]
    assert ex["gaussiansOut"] > 0.95 * lid["splats"] and "viewerEval" not in ex and "denseOccupancy" not in ex
    assert json.loads((out / "scene.json").read_text())["splat"] == {"file": "scene.spz", "lite": "scene_lite.spz", "format": "spz"}
    assert any("held-out" in s for s in lid["skipped"])
