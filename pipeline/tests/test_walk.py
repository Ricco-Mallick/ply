"""Walk mode: the occupancy grid scored with the viewer's own movement rules (plypipe.walkcheck)."""
import cv2
import numpy as np
import pytest

from plypipe import walkcheck
from plypipe.stages import occupancy

from synth import DOOR, PARTITION, two_rooms_model


def _check(model, cams):
    grid, meta = occupancy.grid_from_model(model)
    return grid, meta, walkcheck.check(grid, meta, cams[0, [0, 2]], cams[:, [0, 2]])


def _standable_in_partition(grid, meta):
    ok, xs, zs = walkcheck.standable(grid, meta)
    in_wall = (xs[None, :] > PARTITION[0] - 0.05) & (xs[None, :] < PARTITION[1] + 0.05)
    beside_door = (zs[:, None] < DOOR[0] - 0.05) | (zs[:, None] > DOOR[1] + 0.05)
    return int((ok & in_wall & beside_door).sum())


def test_plain_partition_blocks_and_the_door_passes():
    model, t = two_rooms_model()
    grid, meta, walk = _check(model, t["centers"])
    assert _standable_in_partition(grid, meta) == 0           # no way through the untextured wall
    assert walk["cameras"] == 1.0 and walk["pass"]             # both rooms reachable through the doorway
    assert meta["seenFree"]["cells"] > 0 and meta["pathSegments"] == 1


def test_stray_points_beside_the_walk_do_not_trap_the_walker():
    # two points in the cell next to the walked line, at knee height: the old grid blocked them and the 0.44 m walker
    # could no longer pass
    strays = [(2.0, 1.0, 0.95), (2.02, 1.1, 0.96), (6.0, 1.0, 2.25), (6.02, 0.9, 2.26)]
    model, t = two_rooms_model(strays=strays)
    _, _, walk = _check(model, t["centers"])
    assert walk["cameras"] == 1.0


def test_old_rule_bridged_the_partition():
    """The closed free space of the old grid ran across a plain wall; kept as a record of why it was dropped."""
    model, _ = two_rooms_model()
    grid, meta = occupancy.grid_from_model(model)
    free = (grid == occupancy.FREE).astype(np.uint8)
    closed = cv2.morphologyEx(free, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    bridged = np.where(closed.astype(bool), occupancy.FREE, grid).astype(np.uint8)
    assert _standable_in_partition(bridged, meta) > 0


def test_a_walk_across_furniture_opens_only_that_strip():
    """The trade-off of trusting the walk: where the camera passed over something solid, a 0.2 m strip opens."""
    rng = np.random.default_rng(5)
    block = np.c_[rng.uniform(1, 3, 3000), rng.uniform(0.3, 0.9, 3000), rng.uniform(1, 3, 3000)]
    floor = np.c_[rng.uniform(0, 4, 3000), np.zeros(3000), rng.uniform(0, 4, 3000)]
    cams = np.c_[np.linspace(0.2, 3.8, 25), np.full(25, 1.45), np.full(25, 2.0)]
    grid, meta = occupancy.build_grid(np.vstack([block, floor]), cams)
    assert grid[occupancy.world_to_cell(meta, 2.0, 2.0)] == occupancy.FREE       # under the walk
    assert grid[occupancy.world_to_cell(meta, 2.0, 2.5)] == occupancy.BLOCKED    # 0.5 m beside it
    assert grid[occupancy.world_to_cell(meta, 2.0, 1.5)] == occupancy.BLOCKED


def test_path_breaks_where_the_camera_jumps():
    walk = np.c_[np.linspace(0, 3, 21), np.full(21, 1.45), np.zeros(21)]
    jumped = np.vstack([walk, walk + [0, 0, 6.0]])
    segs = occupancy.path_segments(jumped)
    assert [len(s) for s in segs] == [21, 21]


def test_a_sight_line_carves_only_its_low_part_and_stops_short_of_the_point():
    cam = np.array([[0.05, 1.45, 0.05]])
    pt = np.array([[2.05, 0.0, 0.05]])
    counts = occupancy.seen_free(cam, pt, np.array([0.0, 0.0]), 0.1, (1, 30), below=0.6, stop=0.2)[0]
    # height falls below 0.6 m at x = 0.05 + 2 * (0.85 / 1.45) = 1.22; the line stops 0.2 m short, at 1.85
    assert counts[:12].sum() == 0
    assert counts[13:18].tolist() == [1, 1, 1, 1, 1]           # each cell counted once, not once per sample
    assert counts[19:].sum() == 0


@pytest.mark.parametrize("value, expect", [(occupancy.FREE, True), (occupancy.UNKNOWN, False), (occupancy.BLOCKED, False)])
def test_walkcheck_follows_the_viewer(value, expect):
    grid = np.full((20, 20), value, np.uint8)
    meta = {"cellSize": 0.1, "origin": [0.0, 0.0]}
    ok, _, _ = walkcheck.standable(grid, meta)
    assert bool(ok[20, 20]) is expect                          # the middle of a 2 x 2 m patch
    assert not ok[0, 0]                                         # the rim reaches outside the grid: blocked
