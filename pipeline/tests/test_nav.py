"""The walking grid from depth (stages/occupancy.grid_from_depth, IMPROVEMENT-PLAN 2.3) on two rooms joined by a
0.8 m door through a plain partition 10 cm thick, with depth ray-cast from the solid scene: numpy only, no GPU.
The last tests run export's step around a stand-in for the renderer."""
import json

import cv2
import numpy as np
import pytest

from plypipe import colmap_txt, dense, walkcheck
from plypipe.log import Log
from plypipe.stages import export, occupancy
from plypipe.stages.occupancy import BLOCKED, FREE, UNKNOWN
from plypipe.util import Ctx

from synth import NAV_DOOR, NAV_PARTITION, NAV_ROOMS, NAV_TABLE, cast_depth, depth_views, nav_boxes, nav_model, nav_walk

LO, SHAPE, CELL = np.array([-0.52, -0.53]), (102, 184), occupancy.DEPTH_CELL    # the rooms and half a metre around them
XS, ZS = LO[0] + (np.arange(SHAPE[1]) + 0.5) * CELL, LO[1] + (np.arange(SHAPE[0]) + 0.5) * CELL
RAYS = 6000                                                                     # a view; export's 20,000 changes a dozen cells here


def build(scans, centers):
    return occupancy.grid_from_depth(iter(scans), centers, LO, SHAPE, max_rays=RAYS)


def box(x0, z0, x1, z1):
    """Index of the cells whose centre lies in the rectangle."""
    return np.ix_((ZS > z0) & (ZS < z1), (XS > x0) & (XS < x1))


def cell(x, z):
    return occupancy.world_to_cell({"origin": LO, "cellSize": CELL}, x, z)


@pytest.fixture(scope="module")
def rooms():
    centers, rots = nav_walk()
    views = depth_views(nav_boxes(), centers[::5], rots[::5])                   # every 5th view, as export renders them
    scans = [(p.center, p.points) for p in dense.consistent_points(views)]
    grid, meta = build(scans, centers)
    return centers, scans, grid, meta, occupancy.depth_votes(iter(scans), LO, CELL, SHAPE, max_rays=RAYS)


def standable_in(grid, meta, x0, z0, x1, z1):
    ok, xs, zs = walkcheck.standable(grid, meta)
    return int((ok & (xs[None, :] > x0) & (xs[None, :] < x1) & (zs[:, None] > z0) & (zs[:, None] < z1)).sum())


def standable_in_partition(grid, meta):
    (p0, p1), (z0, z1, _) = NAV_PARTITION, NAV_DOOR
    return standable_in(grid, meta, p0 - 0.05, 0, p1 + 0.05, z0 - 0.05) + standable_in(grid, meta, p0 - 0.05, z1 + 0.05, p1 + 0.05, NAV_ROOMS[1])


def test_the_plain_partition_is_blocked_along_its_whole_length(rooms):
    _, _, grid, meta, _ = rooms
    across = (XS + CELL / 2 > NAV_PARTITION[0]) & (XS - CELL / 2 < NAV_PARTITION[1])  # the cells the wall passes through
    beside = ((ZS + CELL / 2 < NAV_DOOR[0]) | (ZS - CELL / 2 > NAV_DOOR[1])) & (ZS - CELL / 2 > 0) & (ZS + CELL / 2 < NAV_ROOMS[1])
    wall = grid[np.ix_(beside, across)]
    assert beside.sum() >= 60 and across.sum() == 3
    assert (wall == BLOCKED).any(axis=1).all()                                  # seen, in every 5 cm of its length
    assert (wall[:, 1] == UNKNOWN).all()                                        # its inside, which no view can see
    assert standable_in_partition(grid, meta) == 0
    assert meta["source"] == "depth" and meta["cellSize"] == 0.05 and meta["depth"]["views"] == 45


def test_the_sparse_grid_knows_nothing_of_that_partition():
    """The camera solve has no point on a plain wall: no cell of it is blocked, and with a floor as plain the grid is
    the walked corridor, too narrow for the walker."""
    centers, _ = nav_walk()
    grid, meta = occupancy.grid_from_model(nav_model(floor_points=200))
    walk = walkcheck.check(grid, meta, centers[0, [0, 2]], centers[:, [0, 2]])
    assert not walk["pass"] and walk["standableM2"] < 3
    wall = [grid[occupancy.world_to_cell(meta, np.mean(NAV_PARTITION), z)] for z in np.arange(0.3, NAV_ROOMS[1] - 0.25, 0.1)]
    assert BLOCKED not in wall


def test_the_door_is_free_and_the_walker_gets_through_it(rooms):
    centers, _, grid, meta, _ = rooms
    z0, z1, _ = NAV_DOOR
    doorway = box(NAV_PARTITION[0] - 0.1, z0 + 0.1, NAV_PARTITION[1] + 0.1, z1 - 0.1)
    assert grid[doorway].size >= 60 and (grid[doorway] == FREE).all()
    walk = walkcheck.check(grid, meta, centers[0, [0, 2]], centers[:, [0, 2]])   # starts in the first room
    assert walk["cameras"] == 1.0 and walk["pass"] and walk["standableM2"] > 18
    shut = grid.copy()
    shut[box(NAV_PARTITION[0], z0, NAV_PARTITION[1], z1)] = BLOCKED
    assert walkcheck.check(shut, meta, centers[0, [0, 2]], centers[:, [0, 2]])["cameras"] < 0.6  # the door is the only way through


def test_every_camera_position_is_free(rooms):
    centers, _, grid, _, _ = rooms
    assert all(grid[cell(c[0], c[2])] == FREE for c in centers)


def test_a_table_is_blocked(rooms):
    """All of it but the rim away from the cameras: the last pixel before a silhouette is not trusted, so the far
    5 cm of the top is seen by too few views. A cell that only two views saw is unknown, which blocks as well."""
    _, _, grid, meta, _ = rooms
    x0, z0, x1, z1, _ = NAV_TABLE
    top = grid[box(x0 + 0.1, z0 + 0.1, x1 - 0.1, z1 - 0.1)]
    assert top.size >= 150 and (top != FREE).all() and (top == BLOCKED).mean() > 0.97
    assert standable_in(grid, meta, x0, z0, x1, z1) == 0
    assert grid[cell(x0 - 0.3, (z0 + z1) / 2)] == FREE                          # the floor beside it is not


def test_cells_seen_by_fewer_than_three_views_stay_unknown(rooms):
    centers, _, grid, _, votes = rooms
    walked, _ = occupancy.walked_corridor(centers, LO, CELL, SHAPE, 0.2)
    few = (votes["free"] < 3) & (votes["blocked"] < 3) & ~walked
    assert few.sum() > 1000 and (grid[few & (votes["blocked"] > 0)] == UNKNOWN).all()
    assert (grid[few] == FREE).sum() <= 60                                      # only holes ringed by free floor are filled, and
    assert (votes["blocked"][few & (grid == FREE)] == 0).all()                  # none that any view saw something in
    # the same from scratch: views from one spot of a point on the floor 2 m away, the walk beside them
    eye, floor = np.array([0.5, 1.45, 0.5]), np.array([[2.5, 0.0, 0.5]])
    path = np.array([[0.5, 1.45, 0.5], [0.6, 1.45, 0.5]])
    seen = lambda n: build([(eye + [0, 0, 0.01 * k], floor) for k in range(n)], path)[0][cell(1.5, 0.5)]
    assert (seen(2), seen(3)) == (UNKNOWN, FREE)


def test_one_views_stray_point_does_not_block_a_cell(rooms):
    centers, scans, grid, meta, votes = rooms
    spot = (2.0, 2.0)                                                           # open floor in the first room, off the walk
    assert grid[cell(*spot)] == FREE and votes["free"][cell(*spot)] >= 6 and votes["blocked"][cell(*spot)] == 0
    stray = np.array([[spot[0], 0.9, spot[1]]])
    with_stray = lambda n: [(c, np.vstack([p, stray]) if 3 <= k < 3 + n else p) for k, (c, p) in enumerate(scans)]
    one, one_meta = build(with_stray(1), centers)
    assert one[cell(*spot)] == FREE
    assert one_meta["depth"]["seenBlockedByFewer"] == meta["depth"]["seenBlockedByFewer"] + 1  # it was counted, and outvoted
    three, _ = build(with_stray(3), centers)
    assert three[cell(*spot)] == BLOCKED                                        # three views are a thing


def test_a_ray_frees_only_the_part_of_it_a_walker_could_be_in():
    eye = np.array([0.025, 1.45, 0.025])
    row = lambda p, **kw: occupancy.depth_votes([(eye, np.array([p], float))], np.array([0.0, 0.0]), 0.05, (1, 80), **kw)
    floor = row([2.025, 0.0, 0.025])                                            # lands on the floor: its own cell is free too
    assert floor["free"][0, :41].tolist() == [1] * 41 and floor["free"][0, 41:].sum() == 0 and floor["blocked"].sum() == 0
    wall = row([2.025, 1.0, 0.025])                                             # ends on something: that cell is blocked, not free
    assert wall["blocked"][0, 40] == 1 and wall["blocked"].sum() == 1
    assert wall["free"][0, :40].tolist() == [1] * 40 and wall["free"][0, 40:].sum() == 0
    high = row([2.025, 2.45, 0.025])                                            # passes 1.8 m at x = 0.725
    assert high["free"][0, :14].tolist() == [1] * 14 and high["free"][0, 15:].sum() == 0 and high["blocked"].sum() == 0
    below = row([2.025, -1.45, 0.025])                                          # a reflection in the floor: lands at x = 1.025
    assert below["free"][0, :20].tolist() == [1] * 20 and below["free"][0, 21:].sum() == 0 and below["blocked"].sum() == 0
    far = row([3.925, 1.0, 0.025], max_range=2.0)                               # too far to place: free up to 2 m, blocks nothing
    assert far["blocked"].sum() == 0 and far["free"][0, :40].tolist() == [1] * 40 and far["free"][0, 41:].sum() == 0
    assert row([3.925, 1.0, 0.025])["blocked"][0, 78] == 1


def test_a_hole_is_not_filled_over_a_cell_that_was_seen_blocked():
    grid = np.full((9, 9), FREE, np.uint8)
    grid[2:4, 2:4] = grid[5:7, 5:7] = UNKNOWN
    veto = np.zeros((9, 9), bool)
    veto[6, 6] = True
    assert occupancy.fill_holes(grid, 10, veto) == 4
    assert (grid[2:4, 2:4] == FREE).all() and (grid[5:7, 5:7] == UNKNOWN).all()


def test_the_sparse_grid_fills_only_what_the_depth_left_undecided_and_keeps_its_walls(rooms):
    """Each grid where it has evidence: the depth's obstacles stay (the chapel's pews), the sparse grid's walls stay
    (a gap the depth looked through), and floor the depth did not decide on comes from the sparse grid unless a view
    saw something there. A sparse grid that calls the whole footprint free, partition and all (it has no point on
    it), changes nothing outside the rooms and does not open the partition."""
    centers, scans, grid, meta, votes = rooms
    sx = LO[0] + (np.arange(SHAPE[1] // 2) + 0.5) * 0.1
    sz = LO[1] + (np.arange(SHAPE[0] // 2) + 0.5) * 0.1
    sparse = np.full((SHAPE[0] // 2, SHAPE[1] // 2), UNKNOWN, np.uint8)        # 10 cm cells over the same area
    sparse[np.ix_((sz > 0) & (sz < NAV_ROOMS[1]), (sx > 0) & (sx < NAV_ROOMS[0]))] = FREE
    inside = (slice(28, 30), slice(36, 38))                                     # a post in the second room
    sparse[inside] = BLOCKED
    merged, mmeta = occupancy.grid_from_depth(iter(scans), centers, LO, SHAPE, max_rays=RAYS, sparse=sparse)
    fine = cv2.resize(sparse, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_NEAREST)
    walked, _ = occupancy.walked_corridor(centers, LO, CELL, SHAPE, 0.2)
    assert (merged[(grid == BLOCKED)] == BLOCKED).all() and standable_in_partition(merged, mmeta) == 0
    kept = (fine == BLOCKED) & ~walked                                          # the walk wins where it crosses the post
    assert kept.sum() >= 1 and (merged[kept] == BLOCKED).all() and mmeta["depth"]["sparseBlocked"] > 0
    took = (grid == UNKNOWN) & (votes["blocked"] == 0) & (merged == FREE)
    assert mmeta["depth"]["sparseFree"] > 0 and took.sum() > 0
    assert (merged[(grid == UNKNOWN) & (votes["blocked"] > 0) & ~walked] != FREE).all()  # a view saw something there
    footprint = ((ZS > -0.1) & (ZS < NAV_ROOMS[1] + 0.1))[:, None] & ((XS > -0.1) & (XS < NAV_ROOMS[0] + 0.1))[None, :]
    assert np.array_equal(merged[~footprint], grid[~footprint])                 # a sparse cell beyond the rooms changes nothing
    assert mmeta["source"] == "depth" and set(meta) <= set(mmeta)
    plain, _ = occupancy.grid_from_depth(iter(scans), centers, LO, SHAPE, max_rays=RAYS)
    assert np.array_equal(plain, grid)                                          # without a sparse grid, as before


# ------------------------------------------------------------------ export's step, around a stand-in for the GPU

RENDERED = {}


def job(tmp_path, floor_points):
    """A job folder as export finds it: the aligned model, scene.json, and the sparse grid the occupancy stage wrote."""
    colmap_txt.write_model(nav_model(floor_points), tmp_path / "data" / "sparse" / "0")
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={"dense_occupancy": True})
    occupancy.run_stage(ctx)
    return ctx


def ray_cast(ply, views, antialiased=True):
    """What dense.render_views yields, from the solid rooms instead of a splat."""
    for v in views:
        if v.name not in RENDERED:
            RENDERED[v.name] = cast_depth(nav_boxes(), v.K, v.R, v.t, v.width, v.height)
        v.depth, v.alpha = RENDERED[v.name]
        yield v


def test_export_swaps_in_the_grid_from_depth_when_it_walks_at_least_as_well(tmp_path, monkeypatch):
    ctx = job(tmp_path, floor_points=200)                                       # a plain floor: the sparse grid is a corridor
    sparse = json.loads((tmp_path / "occupancy.json").read_text())
    assert sparse["source"] == "sparse" and not sparse["walk"]["pass"]
    monkeypatch.setattr(dense, "render_views", ray_cast)
    rep = export._dense_occupancy(ctx, tmp_path / "scene_clean.ply", 30, True)
    meta = json.loads((tmp_path / "occupancy.json").read_text())
    grid = cv2.imread(str(tmp_path / "occupancy.png"), cv2.IMREAD_GRAYSCALE)
    assert rep["used"] == "depth" and rep["walk"] == meta["walk"] and rep["sparseWalk"] == sparse["walk"] and rep["seconds"] >= 0
    assert meta["source"] == "depth" and meta["walk"]["pass"] and meta["walk"]["standableM2"] > 10 * sparse["walk"]["standableM2"]
    assert rep["views"] == meta["depth"]["views"] == len(dense.training_views(ctx.data_dir, 30))
    assert grid.shape == (meta["height"], meta["width"]) == (2 * sparse["height"], 2 * sparse["width"]) and meta["origin"] == sparse["origin"]
    assert set(sparse) <= set(meta)                                             # nothing a reader of the old file looks for is gone
    assert standable_in_partition(grid, meta) == 0
    assert json.loads((tmp_path / "scene.json").read_text())["occupancy"] == {"image": "occupancy.png", "meta": "occupancy.json"}
    saved = np.load(tmp_path / dense.FILE)
    assert len(saved["points"]) == len(saved["normals"]) == meta["depth"]["points"]["points"] > 50_000
    assert meta["depth"]["pixels"]["kept"] > 0.7 * meta["depth"]["pixels"]["usable"]


def test_export_takes_the_sparse_floor_the_depth_did_not_decide_on(tmp_path, monkeypatch):
    """A floor with a point in every cell: the sparse grid has all of it, the depth only what three views saw. The
    grid written has the depth's partition and the sparse grid's floor."""
    ctx = job(tmp_path, floor_points=5000)
    sparse = json.loads((tmp_path / "occupancy.json").read_text())
    monkeypatch.setattr(dense, "render_views", ray_cast)
    rep = export._dense_occupancy(ctx, tmp_path / "scene_clean.ply", 30, True)
    meta = json.loads((tmp_path / "occupancy.json").read_text())
    grid = cv2.imread(str(tmp_path / "occupancy.png"), cv2.IMREAD_GRAYSCALE)
    assert rep["used"] == "depth" and meta["source"] == "depth" and meta["depth"]["sparseFree"] > 0
    assert rep["walk"]["cameras"] == rep["sparseWalk"]["cameras"] == 1.0 and standable_in_partition(grid, meta) == 0
    assert rep["walk"]["standableM2"] > 0.95 * sparse["walk"]["standableM2"]    # without the sparse floor it stood less (the test before)
    assert not (tmp_path / "occupancy_depth.png").exists()


def test_export_keeps_the_sparse_grid_when_fewer_capture_positions_are_walkable(tmp_path, monkeypatch):
    ctx = job(tmp_path, floor_points=5000)
    before = (tmp_path / "occupancy.png").read_bytes()
    monkeypatch.setattr(dense, "render_views", ray_cast)
    real = walkcheck.check

    def strict(grid, meta, spawn, cams=None, places=None):                      # as if a capture position were cut off
        out = real(grid, meta, spawn, cams, places)
        return {**out, "cameras": out["cameras"] - 0.01} if meta.get("source") == "depth" else out

    monkeypatch.setattr(occupancy.walkcheck, "check", strict)
    rep = export._dense_occupancy(ctx, tmp_path / "scene_clean.ply", 30, True)
    meta = json.loads((tmp_path / "occupancy.json").read_text())
    assert rep["used"] == "sparse" and rep["why"].startswith("fewer capture positions are walkable on it")
    assert (tmp_path / "occupancy.png").read_bytes() == before and meta["source"] == "sparse" and meta["cellSize"] == 0.1
    assert meta["depth"]["rejected"] == rep["why"] and meta["depth"]["walk"] == rep["walk"]
    lost = cv2.imread(str(tmp_path / "occupancy_depth.png"), cv2.IMREAD_GRAYSCALE)   # kept beside it, to look at
    assert lost.shape == (2 * meta["height"], 2 * meta["width"])


def test_a_failing_dense_pass_does_not_fail_the_export(tmp_path, monkeypatch):
    ctx = job(tmp_path, floor_points=200)
    before = {f: (tmp_path / f).read_bytes() for f in ("occupancy.png", "occupancy.json", "scene.json")}

    def out_of_memory(ply, views, antialiased=True):
        yield next(ray_cast(ply, views))
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(dense, "render_views", out_of_memory)
    rep = export._dense_occupancy(ctx, tmp_path / "scene_clean.ply", 30, True)
    assert rep["used"] == "sparse" and rep["why"] == "failed: CUDA out of memory"
    assert {f: (tmp_path / f).read_bytes() for f in before} == before
    logged = [json.loads(ln) for ln in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in logged if "CUDA out of memory" in e["msg"]] == ["warn"]


def test_the_grid_from_depth_is_off_unless_asked_for():
    from plypipe import cli
    run = ["run", "walk.mp4", "--out", "o"]
    assert cli.build_parser().parse_args(run).dense_occupancy is False
    assert cli.build_parser().parse_args([*run, "--dense-occupancy"]).dense_occupancy is True


def test_without_a_gpu_the_flag_changes_nothing_and_says_so(tmp_path):
    ctx = job(tmp_path, floor_points=200)
    before = {f: (tmp_path / f).read_bytes() for f in ("occupancy.png", "occupancy.json", "scene.json")}
    rep = export._dense_occupancy(ctx, tmp_path / "scene_clean.ply", 30, True)
    assert rep["used"] == "sparse" and rep["why"].startswith("not built: needs a GPU")
    assert {f: (tmp_path / f).read_bytes() for f in before} == before and not (tmp_path / dense.FILE).exists()
