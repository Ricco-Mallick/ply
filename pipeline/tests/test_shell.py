"""Measured shell: walls, openings and object boxes recovered from a synthetic room's point cloud."""
import numpy as np
from plypipe import shell

RNG = np.random.default_rng(3)


def room_points(w=6.0, d=4.0, h=2.8, n=9000):
    """Four walls plus a little floor and ceiling noise, in the scene frame."""
    pts = []
    for _ in range(n // 4):  # walls: x = 0 / w, z anywhere; z = 0 / d, x anywhere
        side = RNG.integers(0, 4)
        y = RNG.uniform(0.05, h, 1)
        if side < 2:
            x = 0.0 if side == 0 else w
            pts.append([x + RNG.normal(0, 0.012), y[0], RNG.uniform(0, d)])
        else:
            z = 0.0 if side == 2 else d
            pts.append([RNG.uniform(0, w), y[0], z + RNG.normal(0, 0.012)])
    for _ in range(n // 8):
        pts.append([RNG.uniform(0, w), 0.0 + RNG.normal(0, 0.01), RNG.uniform(0, d)])
    return np.array(pts)


def test_fit_walls_finds_the_four_walls():
    walls = shell.fit_walls(room_points())
    lengths = sorted(round(w["length"], 1) for w in walls)[-4:]
    for got, want in zip(lengths, [4.0, 4.0, 6.0, 6.0]):
        assert abs(got - want) < 0.25, lengths
    # every wall is axis aligned here
    assert all(min(abs(w["angle"]), abs(abs(w["angle"]) - 90)) < 2.5 for w in walls)


def test_wall_normals_point_into_the_room():
    walls = shell.fit_walls(room_points())
    for w in walls:
        mid = np.array([(w["a"][0] + w["b"][0]) / 2, (w["a"][1] + w["b"][1]) / 2])
        n = np.array(w["normal"])
        assert float(n @ (np.array([3.0, 2.0]) - mid)) > 0


def test_doors_attach_to_their_wall():
    walls = shell.fit_walls(room_points())
    boxes = [
        {"id": "door-1", "name": "Door", "cls": "door", "p": [0.02, 1.05, 1.5], "half": [0.45, 1.05, 0.05]},
        {"id": "chair-1", "name": "Chair", "cls": "chair", "p": [3.0, 0.5, 2.0], "half": [0.25, 0.5, 0.25]},
    ]
    found = shell.openings(boxes, walls)
    assert len(found) == 1 and found[0]["kind"] == "door"
    assert 0.8 <= found[0]["width"] <= 1.0


def test_merge_joins_a_wall_split_by_a_doorway():
    walls = [
        {"a": [0, 0], "b": [2, 0], "length": 2.0, "normal": [0, -1], "support": 10},
        {"a": [2.6, 0], "b": [6, 0], "length": 3.4, "normal": [0, -1], "support": 10},
    ]
    merged = shell._merge(walls)
    assert len(merged) == 1 and abs(merged[0]["length"] - 6.0) < 0.01


def test_build_emits_objects_and_openings():
    boxes = [{"id": "w-1", "name": "Window 2", "cls": "window", "p": [3.0, 1.7, 0.03], "half": [0.7, 0.8, 0.05]}]
    out = shell.build(room_points(), 0.0, 2.8, boxes)
    assert out["version"] == 1 and out["ceilingY"] == 2.8
    assert len(out["walls"]) >= 4 and len(out["openings"]) == 1
    assert out["objects"][0]["name"] == "Window 2"
