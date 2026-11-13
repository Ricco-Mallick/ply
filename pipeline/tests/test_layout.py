"""The layout from dense geometry (shell.layout, IMPROVEMENT-PLAN 2.5) on the synthetic rooms of synth.py: the box room, and
two rooms joined by a 0.8 m door through a 10 cm partition. The dense points are what dense.consistent_points keeps of
exactly ray-cast depth, thinned to 2 cm as export saves them; no splat and no GPU. Detections are stand-ins for SAM 3."""
import json

import numpy as np
import pytest

from plypipe import colmap_txt, dense, geometry, layout_eval, shell
from plypipe.log import Log
from plypipe.stages import export
from plypipe.util import Ctx

from synth import NAV_DOOR, NAV_PARTITION, NAV_ROOMS, ROOM, camera_loop, depth_views, look, nav_boxes, nav_model, nav_walk, solid_room


def dense_cloud(boxes, centers, rots, every=1):
    cloud = dense.Cloud()
    for p in dense.consistent_points(depth_views(boxes, centers[::every], rots[::every])):
        cloud.add(p.points, p.normals)
    pts, nrm = cloud.arrays()
    return pts.astype(float), nrm.astype(float)


def turned(points, normals, degrees):
    """The scene turned about the vertical axis through its origin: a room that was not built along the axes."""
    r = geometry.rot_y(np.radians(degrees))
    return points @ r.T, normals @ r.T


def corner_error(walls, corners):
    """For each reference end point, the distance to the nearest wall end point."""
    ends = np.array([w[k] for w in walls for k in ("a", "b")])
    return np.array([np.linalg.norm(ends - c, axis=1).min() for c in corners])


@pytest.fixture(scope="module")
def box_room():
    (c1, r1), (c2, r2) = camera_loop(18), camera_loop(18, inward=True)
    up = np.array([look(a + np.pi + 0.5, 0.5) for a in np.linspace(0, 2 * np.pi, 18, endpoint=False)])  # a lap looking at the ceiling
    centers, rots = np.concatenate([c1, c2, c1]), np.concatenate([r1, r2, up])
    pts, nrm = dense_cloud(solid_room(), centers, rots)
    return pts, nrm, centers


@pytest.fixture(scope="module")
def two_rooms():
    centers, _ = nav_walk()
    stops = centers[::12]  # the walker looks all the way round at every metre, so that every wall is seen, as in a real capture
    views = [(c, look(k * np.pi / 2 + 0.4 + 0.3 * i, 0.1 if k % 2 else -0.1)) for i, c in enumerate(stops) for k in range(4)]
    pts, nrm = dense_cloud(nav_boxes(), np.array([c for c, _ in views]), np.array([r for _, r in views]))
    return pts, nrm, centers


def test_every_wall_of_a_box_room_is_found_with_its_ends_within_ten_centimetres(box_room):
    pts, nrm, path = box_room
    out = shell.layout(pts, nrm, path)
    W, D, H = ROOM
    assert len(out["walls"]) == 4, out["walls"]
    assert corner_error(out["walls"], [(0, 0), (W, 0), (W, D), (0, D)]).max() < 0.10
    assert abs(out["floorY"]) < 0.03 and abs(out["ceilingY"] - H) < 0.03 and out["ceilingSeen"]
    assert out["openings"] == [] and out["validation"]["ok"], out["validation"]
    assert out["source"] == "dense" and all(w["support"] > 0 and w["length"] > 3.5 for w in out["walls"])


def test_wall_normals_point_into_the_room(box_room):
    out = shell.layout(*box_room)
    for w in out["walls"]:
        mid = (np.array(w["a"]) + np.array(w["b"])) / 2
        assert float(np.array(w["normal"]) @ (np.array([ROOM[0] / 2, ROOM[1] / 2]) - mid)) > 0


def test_the_file_keeps_the_fields_the_deployed_viewer_reads(box_room):
    out = json.loads(json.dumps(shell.layout(*box_room)))
    assert {"version", "frame", "floorY", "ceilingY", "walls", "openings", "objects"} <= set(out)
    assert all({"a", "b", "length", "normal", "support", "angle"} <= set(w) for w in out["walls"])


def test_a_room_built_off_the_axes_comes_back_with_square_corners(box_room):
    pts, nrm, path = box_room
    p, n = turned(pts, nrm, 3.0)
    out = shell.layout(p, n, path @ geometry.rot_y(np.radians(3.0)).T)
    assert len(out["walls"]) == 4 and all(w["snapped"] for w in out["walls"])
    turns = sorted(w["angle"] % 90 for w in out["walls"])
    assert turns[-1] - turns[0] < 0.2, turns
    r = geometry.rot_y(np.radians(3.0))
    corners = [(r @ np.array([x, 0, z]))[[0, 2]] for x, z in [(0, 0), (ROOM[0], 0), (ROOM[0], ROOM[1]), (0, ROOM[1])]]
    assert corner_error(out["walls"], corners).max() < 0.10


def test_the_door_is_found_where_the_path_crosses_the_partition_and_nowhere_else(two_rooms):
    pts, nrm, path = two_rooms
    out = shell.layout(pts, nrm, path)
    W, D, _ = NAV_ROOMS
    mid = sum(NAV_PARTITION) / 2
    assert len(out["walls"]) == 5, [(w["a"], w["b"]) for w in out["walls"]]
    assert corner_error(out["walls"], [(0, 0), (W, 0), (W, D), (0, D), (mid, 0), (mid, D)]).max() < 0.10
    doors = [o for o in out["openings"] if o["kind"] == "door"]
    assert len(doors) == 1 and out["openings"] == doors
    wall = out["walls"][doors[0]["wall"]]
    a, b = np.array(wall["a"]), np.array(wall["b"])
    at = a + (b - a) / wall["length"] * doors[0]["u"]
    assert abs(at[0] - mid) < 0.1 and abs(at[1] - sum(NAV_DOOR[:2]) / 2) < 0.1
    assert abs(doors[0]["width"] - (NAV_DOOR[1] - NAV_DOOR[0])) < 0.1 and abs(doors[0]["height"] - NAV_DOOR[2]) < 0.1
    assert doors[0]["source"] == "path" and out["validation"]["ok"], out["validation"]


def test_without_detections_no_window_is_written_and_a_detected_window_is_lifted_onto_its_wall(two_rooms):
    pts, nrm, path = two_rooms
    assert not [o for o in shell.layout(pts, nrm, path)["openings"] if o["kind"] == "window"]
    win = {"cls": "window", "p": [2.0, 1.5, 0.06], "half": [0.5, 0.4, 0.05], "views": 4}
    few = {"cls": "window", "p": [6.0, 1.5, 0.06], "half": [0.5, 0.4, 0.05], "views": 2}
    out = shell.layout(pts, nrm, path, detections=[win, few])
    windows = [o for o in out["openings"] if o["kind"] == "window"]
    assert len(windows) == 1 and abs(windows[0]["width"] - 1.0) < 0.05 and abs(windows[0]["sill"] - 1.1) < 0.05
    assert [r["cls"] for r in out["rejected"]] == ["window"] and "views" in out["rejected"][0]["reason"]


def test_a_door_detected_on_a_solid_wall_is_rejected(two_rooms):
    pts, nrm, path = two_rooms
    door = {"cls": "door", "p": [1.5, 1.0, 0.0], "half": [0.45, 1.0, 0.05], "views": 5}
    out = shell.layout(pts, nrm, path, detections=[door])
    assert [o["kind"] for o in out["openings"]] == ["door"] and out["openings"][0]["source"] == "path"  # only the one the path made
    assert out["rejected"] == [{"cls": "door", "reason": "solid wall", "fill": out["rejected"][0]["fill"]}] and out["rejected"][0]["fill"] > 0.9


def test_validation_fails_a_door_drawn_on_a_solid_wall(box_room):
    pts, nrm, path = box_room
    out = shell.layout(pts, nrm, path)
    out["openings"].append({"kind": "door", "wall": 0, "u": 2.0, "width": 0.9, "sill": 0.0, "height": 2.0})
    check = shell.validate(out, pts, path)
    assert not check["ok"] and "solid wall" in " ".join(check["reasons"]) and check["checks"]["doorsOnSolidWall"] == 1


def test_validation_fails_an_opening_under_half_a_metre_a_low_ceiling_and_a_wall_the_cameras_walked_through(box_room):
    pts, nrm, path = box_room
    out = shell.layout(pts, nrm, path)
    small = dict(out, openings=[{"kind": "window", "wall": 0, "u": 2.0, "width": 0.4, "sill": 1.0, "height": 0.6}])
    assert "under 0.5 m" in " ".join(shell.validate(small, pts, path)["reasons"])
    low = dict(out, ceilingY=1.5)
    assert "ceiling" in " ".join(shell.validate(low, pts, path)["reasons"])
    walked = np.array([[-0.4, 1.45, 2.0], [0.4, 1.45, 2.0]])  # one step through the x = 0 wall (a longer one is read as a jump between clips)
    check = shell.validate(out, pts, np.vstack([path, walked]))
    assert not check["ok"] and check["checks"]["pathCrossings"] >= 1
    stray = dict(out, walls=[dict(out["walls"][0], a=[40.0, 40.0], b=[42.0, 40.0], length=2.0)])
    assert "outside the scanned area" in " ".join(shell.validate(stray, pts, path)["reasons"])


def test_snapping_never_moves_a_wall_end_by_more_than_twenty_centimetres():
    rng = np.random.default_rng(5)
    walls = []
    for _ in range(300):
        a = rng.uniform(-10, 10, 2)
        length, angle = rng.uniform(0.9, 20), np.radians(rng.uniform(0, 180))
        walls.append(shell._wall_fields(a, a + length * np.array([np.cos(angle), np.sin(angle)]), np.array([0.0, 1.0]), 100))
    before = [(np.array(w["a"]), np.array(w["b"])) for w in walls]
    shell.snap_walls(walls, 12.0)
    assert any(w["snapped"] for w in walls) and not all(w["snapped"] for w in walls)
    for w, (a, b) in zip(walls, before):
        assert max(np.linalg.norm(np.array(w["a"]) - a), np.linalg.norm(np.array(w["b"]) - b)) <= 0.2 + 0.002
        if w["snapped"]:
            assert abs((w["angle"] - 12.0 + 45) % 90 - 45) < 0.1


def test_a_long_wall_a_few_degrees_off_is_left_alone_and_a_short_one_is_turned():
    long = shell._wall_fields(np.array([0.0, 0.0]), 15 * np.array([np.cos(np.radians(3)), np.sin(np.radians(3))]), np.array([0.0, 1.0]), 1)
    short = shell._wall_fields(np.array([0.0, 0.0]), 3 * np.array([np.cos(np.radians(3)), np.sin(np.radians(3))]), np.array([0.0, 1.0]), 1)
    far = shell._wall_fields(np.array([0.0, 0.0]), 3 * np.array([np.cos(np.radians(8)), np.sin(np.radians(8))]), np.array([0.0, 1.0]), 1)
    shell.snap_walls([long, short, far], 0.0)
    assert (long["snapped"], short["snapped"], far["snapped"]) == (False, True, False)
    assert abs(short["angle"]) < 0.1 and abs(long["angle"] - 3.0) < 0.1


# ------------------------------------------------------------------ export

def write_job(tmp_path, pts, nrm, path_model, with_sparse=True):
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    np.savez_compressed(tmp_path / dense.FILE, points=pts.astype(np.float32), normals=nrm.astype(np.float16), voxel=np.float32(0.02))
    colmap_txt.write_model(path_model, ctx.data_dir / "sparse" / "0")
    if with_sparse:
        (tmp_path / "shell.json").write_text(json.dumps(shell.build(path_model.xyz, 0.0, 2.8, [])))
    return ctx


def test_export_ships_the_dense_layout_when_it_validates_and_keeps_the_sparse_one_when_it_does_not(two_rooms, box_room, tmp_path):
    pts, nrm, _ = two_rooms
    model = nav_model()
    (tmp_path / "ok").mkdir()
    ctx = write_job(tmp_path / "ok", pts, nrm, model)
    sparse = json.loads((ctx.out / "shell.json").read_text())
    got = export._dense_layout(ctx)
    now = json.loads((ctx.out / "shell.json").read_text())
    assert got["used"] == "dense" and now["source"] == "dense" and now["validation"]["ok"] and len(now["openings"]) == 1
    assert json.loads((ctx.out / "shell_dense.json").read_text())["validation"]["ok"] and "walls" in sparse

    # the box room's points and the two-room walk: the cameras walk through a wall that is solid in the points
    (tmp_path / "bad").mkdir()
    ctx = write_job(tmp_path / "bad", box_room[0], box_room[1], model)
    before = json.loads((ctx.out / "shell.json").read_text())
    got = export._dense_layout(ctx)
    now = json.loads((ctx.out / "shell.json").read_text())
    assert got["used"] == "sparse" and ("crosses wall" in got["why"] or "outside" in got["why"])
    assert now["walls"] == before["walls"] and now.get("source", "sparse") == "sparse" and now["denseLayout"]["ok"] is False
    failed = json.loads((ctx.out / "shell_dense.json").read_text())
    assert failed["validation"]["ok"] is False and failed["validation"]["reasons"]


def test_export_leaves_a_job_without_dense_points_or_with_the_option_off_exactly_as_it_was(two_rooms, tmp_path):
    ctx = write_job(tmp_path, *two_rooms[:2], nav_model())
    (tmp_path / dense.FILE).rename(tmp_path / "elsewhere.npz")
    text = (tmp_path / "shell.json").read_text()
    assert export._dense_layout(ctx) is None and (tmp_path / "shell.json").read_text() == text
    (tmp_path / "elsewhere.npz").rename(tmp_path / dense.FILE)
    ctx.opts["dense_layout"] = False
    assert export._dense_layout(ctx) is None and (tmp_path / "shell.json").read_text() == text and not (tmp_path / "shell_dense.json").exists()


TWO_ROOM_PLAN = {"walls": [{"a": [0, 0], "b": [8.1, 0]}, {"a": [8.1, 0], "b": [8.1, 4]}, {"a": [8.1, 4], "b": [0, 4]}, {"a": [0, 4], "b": [0, 0]},
                           {"a": [4.05, 0], "b": [4.05, 4]}], "doors": [{"p": [4.05, 2.1]}]}


def test_the_layout_score_passes_the_ship_rule_on_the_two_rooms_and_fails_it_with_a_missing_door_or_a_false_one(two_rooms):
    out = shell.layout(*two_rooms)
    got = layout_eval.score(out, TWO_ROOM_PLAN, rooms=2)
    assert got["ships"] and got["medianEndpointCm"] <= 10 and got["found"] == got["real"] == 1 and got["false"] == 0, got
    missing = layout_eval.score(dict(out, openings=[]), TWO_ROOM_PLAN, rooms=2)
    assert not missing["ships"] and missing["found"] == 0
    extra = dict(out["openings"][0], wall=0, u=3.0)
    assert layout_eval.score(dict(out, openings=out["openings"] + [extra] * 3), TWO_ROOM_PLAN, rooms=1)["false"] == 3
    assert not layout_eval.score(dict(out, openings=out["openings"] + [extra] * 3), TWO_ROOM_PLAN, rooms=1)["ships"]
    assert not layout_eval.score(dict(out, walls=out["walls"][:2]), TWO_ROOM_PLAN)["ships"]


def test_wall_flatness_is_under_a_centimetre_on_exact_depth_and_shows_a_bump_in_a_wall(box_room):
    pts, nrm, _ = box_room
    out = shell.layout(pts, nrm, box_room[2])
    flat = layout_eval.flatness_summary(layout_eval.wall_flatness(pts, out))
    assert flat["walls"] == 4 and flat["worstRmsCm"] < 1.0, flat
    rng = np.random.default_rng(3)
    wavy = pts.copy()
    on_wall = np.abs(wavy[:, 0]) < 0.15
    wavy[on_wall, 0] += 0.06 * np.sin(wavy[on_wall, 2] * 4) + rng.normal(0, 0.01, on_wall.sum())
    bad = layout_eval.wall_flatness(wavy, out)
    assert max(w["rms"] for w in bad if w["rms"] is not None) > 0.03
