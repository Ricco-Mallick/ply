# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""plypipe.people: masks on disk, the warp into undistorted views, the trainer's Dataset wrapper, and COLMAP honouring
the masks. The detector is a stand-in: SAM 3 itself only runs on the GPU image."""

import json
import types

import cv2
import numpy as np
import pytest

from plypipe import people


def _frames(tmp_path, n=4, w=160, h=120):
    d = tmp_path / "frames"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(d / f"{i:06d}.jpg"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return d


def test_mask_frames_writes_colmap_masks_black_on_people_and_drops_mostly_person_frames(tmp_path):
    frames = _frames(tmp_path)

    def detect(rgb):
        h, w = rgb.shape[:2]
        m = np.zeros((h, w), bool)
        i = detect.calls
        detect.calls += 1
        if i == 1:
            m[40:80, 60:90] = True       # someone in the middle
        if i == 3:
            m[:, :] = True               # someone right in front of the lens
        return m
    detect.calls = 0

    rep = people.mask_frames(frames, tmp_path, detect, max_share=0.45)
    assert rep["frames"] == 4 and rep["framesWithPeople"] == 2
    assert [d["frame"] for d in rep["dropped"]] == ["000003.jpg"]
    assert not (frames / "000003.jpg").exists() and (tmp_path / people.DROPPED / "000003.jpg").exists()
    m = cv2.imread(str(tmp_path / "masks" / "000001.jpg.png"), cv2.IMREAD_GRAYSCALE)   # COLMAP's <image>.png naming
    assert m[60, 75] == 0 and m[5, 5] == 255                                             # black on the person
    assert m[40 - 2, 75] == 0                                                            # grown past the box edge
    assert (cv2.imread(str(tmp_path / "masks" / "000000.jpg.png"), cv2.IMREAD_GRAYSCALE) == 255).all()
    assert not (tmp_path / "masks" / "000003.jpg.png").exists()
    assert people.cached(tmp_path, frames)["framesWithPeople"] == 2
    assert json.loads((tmp_path / "masks" / "people.json").read_text())["prompt"] == "person"


def test_cached_refuses_when_a_frame_has_no_mask(tmp_path):
    frames = _frames(tmp_path, n=2)
    people.mask_frames(frames, tmp_path, lambda rgb: np.zeros(rgb.shape[:2], bool))
    cv2.imwrite(str(frames / "000009.jpg"), np.zeros((120, 160, 3), np.uint8))
    assert people.cached(tmp_path, frames) is None


def test_warp_is_identity_for_a_pinhole_source():
    keep = np.ones((90, 120), bool)
    keep[30:50, 20:70] = False
    out = people.warp_mask(keep, "PINHOLE", [100, 100, 60, 45], [100, 100, 60, 45], 120, 90)
    assert (out == keep).all()


@pytest.mark.parametrize("model,params,dist", [
    ("SIMPLE_RADIAL", [400.0, 320.0, 240.0, -0.12], (-0.12, 0, 0, 0)),
    ("OPENCV", [410.0, 400.0, 318.0, 242.0, -0.1, 0.03, 0.001, -0.002], (-0.1, 0.03, 0.001, -0.002)),
])
def test_warp_puts_a_person_where_opencv_undistorts_it(model, params, dist):
    """Independent check: a blob near the corner of the distorted frame must land where cv2.undistortPoints says."""
    w, h = 640, 480
    keep = np.ones((h, w), bool)
    ud, vd = 560, 400                                  # cv2 pixel coords (centres at integers)
    keep[vd - 6:vd + 7, ud - 6:ud + 7] = False
    if model == "SIMPLE_RADIAL":
        K = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]])
    else:
        K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    dst = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    out = people.warp_mask(keep, model, params, dst, w, h)
    # COLMAP coords are cv2 + 0.5 on both sides; with the same K they cancel
    p = cv2.undistortPoints(np.array([[[ud + 0.5, vd + 0.5]]], np.float64), K, np.array(dist), P=K)[0, 0] - 0.5
    x, y = int(round(p[0])), int(round(p[1]))
    assert not out[y, x], "the person did not move with the lens"
    assert out[vd, ud] or (abs(x - ud) + abs(y - vd) < 3)
    assert out.mean() > 0.99


def test_undistort_masks_uses_each_images_own_camera(tmp_path):
    (tmp_path / "masks").mkdir()
    keep = np.ones((120, 160), bool)
    keep[50:70, 70:90] = False
    people.write_mask(tmp_path / "masks" / "a.jpg.png", ~keep)
    people.write_mask(tmp_path / "masks" / "b.jpg.png", np.zeros((120, 160), bool))
    cam = lambda model, params, w=160, h=120: types.SimpleNamespace(model=types.SimpleNamespace(name=model), params=params, width=w, height=h)
    im = lambda name, cid: types.SimpleNamespace(name=name, camera_id=cid)
    src = types.SimpleNamespace(cameras={1: cam("SIMPLE_RADIAL", [150, 80, 60, 0.0])}, images={1: im("a.jpg", 1), 2: im("b.jpg", 1)})
    dst = types.SimpleNamespace(cameras={7: cam("PINHOLE", [150, 150, 70, 55], 140, 110), 8: cam("PINHOLE", [150, 150, 80, 60])},
                                images={1: im("a.jpg", 7), 2: im("b.jpg", 8), 3: im("c.jpg", 8)})
    r = people.undistort_masks(tmp_path, src, dst, tmp_path / "dense" / "masks")
    assert r == {"masks": 2, "withoutMask": 1}
    a = people.read_keep(tmp_path / "dense" / "masks" / "a.jpg.png")
    assert a.shape == (110, 140)                    # the undistorted camera's size, not the frame's
    assert not a[60 - 5, 80 - 10] and a[5, 5]       # the person moved by the crop offset (10, 5)
    assert people.read_keep(tmp_path / "dense" / "masks" / "b.jpg.png").all()


def _fake_st(names, h=20, w=30, with_mask=False, points=None):
    torch = pytest.importorskip("torch")

    class Dataset:
        def __init__(self):
            self.parser = types.SimpleNamespace(image_names=names)
            self.indices = list(range(len(names)))

        def __getitem__(self, i):
            d = {"image": torch.zeros(h, w, 3)}
            if with_mask:
                m = torch.ones(h, w, dtype=torch.bool)
                m[:, :2] = False
                d["mask"] = m
            if points is not None:
                d["points"], d["depths"] = points.clone(), torch.arange(len(points), dtype=torch.float32)
            return d
    return types.SimpleNamespace(Dataset=Dataset), torch


def test_install_masks_people_joins_gsplats_mask_and_drops_depth_on_them(tmp_path):
    torch = pytest.importorskip("torch")
    pts = torch.tensor([[x + 0.0, 10.0] for x in range(30)])
    st, torch = _fake_st(["000001.jpg", "000002.jpg", "000003.jpg"], with_mask=True, points=pts)
    person = np.zeros((20, 30), bool)
    person[:, 20:] = True
    people.write_mask(tmp_path / "000001.jpg.png", person)
    people.write_mask(tmp_path / "000002.jpg.png", np.zeros((40, 60), bool))   # wrong size: trains whole
    stats = people.install(st, tmp_path, log=lambda *_: None)
    ds = st.Dataset()
    d = ds[0]
    assert d["mask"].dtype == torch.bool and d["mask"].shape == (20, 30)
    assert not d["mask"][:, 20:].any() and not d["mask"][:, :2].any() and d["mask"][:, 2:20].all()
    assert len(d["points"]) == 18 and float(d["points"][:, 0].max()) == 19   # only depth samples off the person and gsplat's mask
    assert ds[1]["mask"][:, 5:].all()       # skipped: gsplat's own mask untouched
    assert ds[2]["mask"][:, 5:].all()       # no mask file
    assert stats["applied"] == 1 and stats["skipped"] == 1


def test_install_needs_masks(tmp_path):
    st, _ = _fake_st(["a.jpg"])
    with pytest.raises(ValueError):
        people.install(st, tmp_path)


def test_hide_only_touches_people():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    keep = np.ones((80, 120), bool)
    keep[20:60, 40:80] = False
    out = people.hide(img, keep)
    assert (out[keep] == img[keep]).all()
    assert np.abs(out[~keep].astype(int) - img[~keep].astype(int)).mean() > 20


def test_colmap_finds_no_features_on_people(tmp_path):
    """The real pycolmap: our file names and polarity are what COLMAP's mask_path expects."""
    pycolmap = pytest.importorskip("pycolmap")
    from plypipe.stages import sfm
    imgs = tmp_path / "frames"
    imgs.mkdir()
    rng = np.random.default_rng(3)
    img = cv2.resize(rng.integers(0, 255, (60, 80), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    for i in range(2):
        cv2.imwrite(str(imgs / f"{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    person = np.zeros((480, 640), bool)
    person[:, :320] = True
    masks = tmp_path / "masks"
    people.write_mask(masks / "000000.jpg.png", person)
    people.write_mask(masks / "000001.jpg.png", np.zeros((480, 640), bool))
    db = tmp_path / "db.db"
    sfm._extract(pycolmap, db, imgs, "sift", pycolmap.Device("cpu"), masks=masks)
    d = pycolmap.Database.open(db)
    try:
        by = {im.name: im.image_id for im in d.read_all_images()}
        k0 = np.asarray(d.read_keypoints(by["000000.jpg"]))
        k1 = np.asarray(d.read_keypoints(by["000001.jpg"]))
    finally:
        d.close()
    assert len(k1) > 50 and (k1[:, 0] < 320).any()
    assert len(k0) > 20 and (k0[:, 0] >= 318).all(), "features were found on the masked person"


def test_review_sheet_blurs_people_and_skips_frames_without_them(tmp_path):
    frames = _frames(tmp_path, n=3)

    def detect(rgb):
        m = np.zeros(rgb.shape[:2], bool)
        if detect.n == 1:
            m[30:90, 40:100] = True
        detect.n += 1
        return m
    detect.n = 0
    people.mask_frames(frames, tmp_path, detect)
    sheet = people.review_sheet(tmp_path)
    img = cv2.imread(str(sheet))
    assert img.shape[1] == 320                        # one frame had someone in it: one tile
    assert people.review_sheet(_empty := tmp_path / "x") is None
