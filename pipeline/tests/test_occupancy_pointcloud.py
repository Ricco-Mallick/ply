"""The walking grid from a scan's own returns (stages/occupancy.grid_from_pointcloud, IMPROVEMENT-PLAN phase 6): a
scanned room's floor is free, its walls and furniture are blocked, what the scanner never saw is unknown, and a cell
whose column the beam went over to the ceiling is free even where no return came off the floor. No camera pose is
needed: the grid is the cloud's alone."""

import numpy as np
import pytest

from plypipe import walkcheck
from plypipe.stages import occupancy
from plypipe.stages.occupancy import BLOCKED, FREE, UNKNOWN

from lidar_scene import scanned_room
from synth import ROOM, SOFA, TABLE

WALLS = (2, 3, 4, 5)
FLOOR, CEILING = 0, 1


@pytest.fixture(scope="module")
def scan():
    return scanned_room()


@pytest.fixture(scope="module")
def grid(scan):
    return occupancy.grid_from_pointcloud(scan["points"])


def clear_of_furniture(q: np.ndarray, gap: float = 0.3) -> np.ndarray:
    """Floor points at least `gap` from every box's footprint: a cell on a box's edge legitimately holds both."""
    out = np.ones(len(q), bool)
    for x0, z0, x1, z1, _ in (TABLE, SOFA):
        out &= ~((q[:, 0] > x0 - gap) & (q[:, 0] < x1 + gap) & (q[:, 2] > z0 - gap) & (q[:, 2] < z1 + gap))
    return out


def clear_of_walls(q: np.ndarray, gap: float = 0.3) -> np.ndarray:
    W, D, _ = ROOM
    return (q[:, 0] > gap) & (q[:, 0] < W - gap) & (q[:, 2] > gap) & (q[:, 2] < D - gap)


def cells_of(meta, q):
    col = np.floor((q[:, 0] - meta["origin"][0]) / meta["cellSize"]).astype(int)
    row = np.floor((q[:, 2] - meta["origin"][1]) / meta["cellSize"]).astype(int)
    return col, row


def test_the_grid_is_the_scans_alone_and_says_so(grid):
    g, meta = grid
    assert meta["source"] == "pointcloud"
    assert meta["cellSize"] == occupancy.DEPTH_CELL == 0.05
    assert (meta["stats"]["free"] + meta["stats"]["blocked"] + meta["stats"]["unknown"]) == meta["width"] * meta["height"]
    assert g.shape == (meta["height"], meta["width"]) and set(np.unique(g)) <= {FREE, BLOCKED, UNKNOWN}
    assert meta["stats"]["free"] > 0 and meta["stats"]["blocked"] > 0 and meta["stats"]["unknown"] > 0
    assert meta["pointcloud"]["floorCells"] > 0


def test_the_walls_block_and_the_floor_is_free(grid, scan):
    g, meta = grid
    col, row = cells_of(meta, scan["points"])
    q, face = scan["points"], scan["face"]
    wall = (np.isin(face, WALLS) & (q[:, 1] > 0.3) & (q[:, 1] < 1.8))
    assert (g[row[wall], col[wall]] == BLOCKED).mean() > 0.97
    inner = (face == FLOOR) & clear_of_furniture(q) & clear_of_walls(q)
    assert inner.sum() > 100 and (g[row[inner], col[inner]] == FREE).mean() > 0.95


def test_the_table_and_the_sofa_block(grid, scan):
    g, meta = grid
    col, row = cells_of(meta, scan["points"])
    q = scan["points"]
    boxes = (scan["face"] >= 6) & (q[:, 1] > 0.3)
    assert boxes.sum() > 100 and (g[row[boxes], col[boxes]] == BLOCKED).mean() > 0.97


def test_a_cell_the_beam_went_over_to_the_ceiling_is_free_though_its_floor_is_dark():
    """One slab of floor gives no return at all (dark, glossy), the walls and the ceiling do: the beam crossed that cell
    on its way to the ceiling, so nothing stands in it and it is floor. Without a ceiling to reach, it stays unknown."""
    cell, W, D, H = 0.05, 3.0, 2.0, 2.6
    rng = np.random.default_rng(0)
    lo = np.zeros(2)
    shape = (int(np.ceil((D + 1.0) / cell)), int(np.ceil((W + 1.0) / cell)))
    floor = np.stack([rng.uniform(0, W, 40_000), np.zeros(40_000), rng.uniform(0, D, 40_000)], 1)
    dark = (floor[:, 0] > 1.2) & (floor[:, 0] < 2.0)  # a strip with no floor returns
    walls = np.concatenate([np.stack([np.full(20_000, x), rng.uniform(0.2, 1.8, 20_000), rng.uniform(0, D, 20_000)], 1)
                            for x in (0.0, W)] + [np.stack([rng.uniform(0, W, 20_000), rng.uniform(0.2, 1.8, 20_000),
                                                             np.full(20_000, z)], 1) for z in (0.0, D)])
    seen = np.concatenate([floor[~dark], walls])
    g, meta = occupancy.grid_from_pointcloud(seen, lo=lo, shape=shape, cell=cell, top=None)
    col, row = cells_of(meta, np.array([[1.6, 0.0, 1.0]]))
    assert g[row, col] == UNKNOWN, "no floor return and no ceiling rule: nothing says a walker can stand there"
    gx, gz = np.meshgrid(np.arange(0.1, W - 0.1, 0.04), np.arange(0.1, D - 0.1, 0.04))
    ceiling_returns = np.stack([gx.ravel(), np.full(gx.size, H), gz.ravel()], 1)   # the ceiling, as a scan sees it
    g2, meta2 = occupancy.grid_from_pointcloud(np.concatenate([seen, ceiling_returns]), lo=lo, shape=shape, cell=cell, top="auto")
    assert meta2["pointcloud"]["ceilingY"] == pytest.approx(H, abs=0.05)
    assert g2[row, col] == FREE, "the beam reached the ceiling over that cell: nothing stands in it"
    assert meta2["pointcloud"]["seenThroughCells"] > 0


def test_a_ceiling_under_head_height_is_not_a_ceiling():
    """An outdoor scan, or one whose upper layer is 2 m up: seeing 'through to it' would free cells for the wrong reason,
    so the rule is off and only floor returns free anything."""
    rng = np.random.default_rng(1)
    floor = np.stack([rng.uniform(0, 2, 5_000), np.zeros(5_000), rng.uniform(0, 2, 5_000)], 1)
    low_roof = np.array([[0.5, 2.1, 0.5], [1.5, 2.1, 1.5]])  # 2.1 m up: 0.3 m over the band's top
    g, meta = occupancy.grid_from_pointcloud(np.concatenate([floor, low_roof]), cell=0.05)
    assert meta["pointcloud"]["ceilingY"] is None and meta["pointcloud"]["seenThroughCells"] == 0


def test_the_walker_stands_in_the_room_and_not_in_the_furniture(grid, scan):
    g, meta = grid
    ok, xs, zs = walkcheck.standable(g, meta)
    assert ok.sum() * walkcheck.STEP ** 2 > 4.0, "there is somewhere to stand in a 6 x 4 m room with its middle clear"
    walk = walkcheck.check(g, meta, (3.0, 2.0), np.array([[0.5, 0.5], [3.0, 2.0], [5.5, 3.5]]))
    assert walk["standableM2"] > 2.0
    # a table 1 m across with a 0.75 m top: the walker is not standable on it
    for box in (TABLE, SOFA):
        x0, z0, x1, z1, _ = box
        for x in np.linspace(x0 + 0.1, x1 - 0.1, 3):
            for z in np.linspace(z0 + 0.1, z1 - 0.1, 3):
                i = int(round((x - xs[0]) / walkcheck.STEP))
                j = int(round((z - zs[0]) / walkcheck.STEP))
                assert not ok[j, i], f"standable over the box at {x:.2f}, {z:.2f}"


def test_the_walls_close_the_room(grid, scan):
    """Every free cell is inside the room's own outline: a scan frees nothing beyond the wall it saw."""
    g, meta = grid
    W, D, _ = ROOM
    xs = meta["origin"][0] + (np.arange(meta["width"]) + 0.5) * meta["cellSize"]
    zs = meta["origin"][1] + (np.arange(meta["height"]) + 0.5) * meta["cellSize"]
    outside = (xs[None, :] < -0.15) | (xs[None, :] > W + 0.15) | (zs[:, None] < -0.15) | (zs[:, None] > D + 0.15)
    assert not ((g == FREE) & outside).any()
