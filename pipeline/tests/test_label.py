"""SAM 3 detections -> 3D objects, without the model: masks are point-id sets, as the stage builds them."""
import numpy as np

from plypipe.stages.label import lift, places_from


def test_lift_merges_views_and_boxes_the_object():
    rng = np.random.default_rng(0)
    chair = rng.uniform([1, 0, 1], [1.5, 0.9, 1.5], (60, 3))        # points of one chair
    door = rng.uniform([-2, 0, 3], [-1.1, 2.0, 3.05], (80, 3))      # and of a door
    stray = np.array([[9.0, 9.0, 9.0]])                               # a mismatched point inside one mask
    xyz = {i: p for i, p in enumerate(np.vstack([chair, door, stray]))}
    dets = [{"cls": "chair", "score": .9, "frame": "a", "ids": set(range(0, 40)) | {140}},
            {"cls": "chair", "score": .8, "frame": "b", "ids": set(range(20, 60))},
            {"cls": "door", "score": .95, "frame": "a", "ids": set(range(60, 140))},
            {"cls": "chair", "score": .6, "frame": "c", "ids": set(range(60, 70))},     # same pixels, wrong class, one frame
            {"cls": "lamp", "score": .55, "frame": "b", "ids": {1, 2, 3}}]           # too few points
    objs = lift(dets, xyz)
    cls = sorted(o["cls"] for o in objs)
    assert cls == ["chair", "door"], objs  # the stray chair-on-door detection and the 3-point lamp are dropped
    ch = next(o for o in objs if o["cls"] == "chair")
    assert ch["frames"] == 2 and np.allclose(ch["center"], [1.25, .45, 1.25], atol=.08) and max(ch["half"]) < .4  # the 9 m outlier is ignored
    places = places_from(objs)
    door = next(p for p in places if p["ai"]["cls"] == "door")
    assert door["cat"] == "Exits" and door["pinned"] and door["name"] == "Door"


def test_names_are_numbered_per_class():
    objs = [{"cls": "chair", "conf": .9, "frames": 2, "points": 9, "center": [0, 0, i], "half": [.2, .2, .2]} for i in range(3)]
    assert [p["name"] for p in places_from(objs)] == ["Chair", "Chair 2", "Chair 3"]
