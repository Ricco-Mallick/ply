"""Floors and stairs from the camera heights of one walk."""
import numpy as np

from plypipe import levels


def walk_two_floors(rng):
    """Ground floor loop at eye 1.45, 12 stair frames rising 3.2 m, upper floor loop, back down the same stairs."""
    t = np.linspace(0, 2 * np.pi, 120)
    ground = np.c_[3 * np.cos(t), 1.45 + rng.normal(0, .05, t.size), 2 * np.sin(t)]
    up = np.c_[np.linspace(3, 3, 12), 1.45 + np.linspace(.27, 2.93, 12), np.linspace(0, 3, 12)]
    upper = np.c_[3 + 2 * np.sin(t), 4.65 + rng.normal(0, .05, t.size), 3 + 2 * np.cos(t) - 2]
    down = up[::-1]
    return np.vstack([ground, up, upper, down, ground[:30]])


def test_two_floors_and_one_staircase():
    rng = np.random.default_rng(0)
    cams = walk_two_floors(rng)
    pts = np.c_[rng.uniform(-3, 5, 4000), np.r_[np.zeros(2000), np.full(2000, 3.21)], rng.uniform(-2, 4, 4000)]
    got = levels.find(cams, pts)
    assert [round(l["floorY"], 1) for l in got["levels"]] == [0.0, 3.2]
    assert len(got["stairs"]) == 1, got["stairs"]
    s = got["stairs"][0]
    assert (s["from"], s["to"]) == (0, 1)
    ys = [p[1] for p in s["path"]]
    assert ys[0] == 0.0 and abs(ys[-1] - 3.21) < 0.03 and ys == sorted(ys)
    assert sum(v == -1 for v in got["frameLevel"]) == 16   # stair frames within LEVEL_BAND of a floor count as on it


def test_one_floor_is_one_level_and_no_stairs():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 6, 200)
    cams = np.c_[np.cos(t), 1.4 + rng.normal(0, .1, t.size), np.sin(t)]
    got = levels.find(cams)
    assert len(got["levels"]) == 1 and got["stairs"] == [] and got["levels"][0]["floorY"] == 0.0


def test_occupancy_writes_a_grid_per_floor_and_the_stairs(tmp_path):
    import json
    import cv2
    from plypipe import colmap_txt
    from plypipe.log import Log
    from plypipe.stages import occupancy
    from plypipe.util import Ctx
    rng = np.random.default_rng(2)
    cams = walk_two_floors(rng)
    # two 8 x 6 m floors, 3.2 m apart; a table on the ground floor, a wall stub on the upper one
    def slab(y, n=6000): return np.c_[rng.uniform(-4, 4, n), np.full(n, y), rng.uniform(-3, 3, n)]
    walls = np.vstack([np.c_[rng.uniform(-4, 4, 3000), rng.uniform(0, 6.4, 3000), np.full(3000, z)] for z in (-3.0, 3.0)])
    table = np.c_[rng.uniform(-1.5, -0.5, 800), rng.uniform(0.7, 0.75, 800), rng.uniform(-0.5, 0.5, 800)]
    stub = np.c_[rng.uniform(-2.55, -2.45, 800), 3.2 + rng.uniform(0.2, 1.8, 800), rng.uniform(-1.0, 1.0, 800)]
    pts = np.vstack([slab(0.0), slab(3.2), walls, table, stub])
    n, k = len(pts), len(cams)
    images = [colmap_txt.Image(i + 1, np.array([1.0, 0, 0, 0]), -c, 1, f"{i:06d}.jpg", "") for i, c in enumerate(cams)]
    tails = [f"128 128 128 0.5 {1 + i % k} 0 {1 + (i + 1) % k} 1 {1 + (i + 2) % k} 2" for i in range(n)]
    model = colmap_txt.Model(["# Camera list", "1 PINHOLE 1600 1200 1200 1200 800 600"], images, np.arange(1, n + 1), pts,
                             np.full((n, 3), 128, np.uint8), np.full(n, 0.5), np.full(n, 3, np.int32), tails)
    colmap_txt.write_model(model, tmp_path / "data" / "sparse" / "0")
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    rep = occupancy.run_stage(ctx)
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert [round(L["floorY"], 1) for L in scene["levels"]] == [0.0, 3.2] and len(scene["stairs"]) == 1
    assert scene["occupancy"]["image"] == "occupancy.png" and scene["levels"][1]["occupancy"]["image"] == "occupancy_L1.png"
    g0 = cv2.imread(str(tmp_path / "occupancy.png"), cv2.IMREAD_GRAYSCALE); m0 = json.loads((tmp_path / "occupancy.json").read_text())
    g1 = cv2.imread(str(tmp_path / "occupancy_L1.png"), cv2.IMREAD_GRAYSCALE); m1 = json.loads((tmp_path / "occupancy_L1.json").read_text())
    at = lambda g, m, x, z: int(g[occupancy.world_to_cell(m, x, z)])
    assert at(g0, m0, -1.0, 0.0) == occupancy.BLOCKED       # the table, ground floor only
    assert at(g1, m1, -1.0, 0.0) != occupancy.BLOCKED       # nothing there upstairs
    assert at(g1, m1, -2.5, 0.0) == occupancy.BLOCKED       # the stub wall, upper floor only
    assert at(g0, m0, -2.5, 0.0) != occupancy.BLOCKED
    assert rep["stairs"] == 1 and len(rep["levels"]) == 2
