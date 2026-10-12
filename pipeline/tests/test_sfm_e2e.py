"""End-to-end on CPU: render a textured synthetic room from a walked loop, then run the real
frames -> sfm (pycolmap, SIFT) -> align -> occupancy stages through the CLI and compare with
ground truth. Skipped when pycolmap is not installed (it is in the GPU env)."""

import json

import cv2
import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from plypipe import geometry  # noqa: E402
from plypipe.cli import main  # noqa: E402

from synth import ROOM, TABLE, camera_loop  # noqa: E402

W_IMG, H_IMG, F = 960, 720, 700.0


class ValueNoise:
    """Non-repeating multi-octave value noise per surface: periodic textures make SfM match the
    wrong wall and fold the loop."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.L = [rng.random((300, 300, 3)) for _ in range(3)]

    def __call__(self, u, v):
        out = np.zeros(u.shape + (3,))
        for o, cell in enumerate((0.5, 0.16, 0.05)):
            L = self.L[o]
            gu, gv = u / cell + 7.3, v / cell + 3.1
            i = np.floor(gu).astype(int) % 299
            j = np.floor(gv).astype(int) % 299
            fu, fv = (gu % 1)[..., None], (gv % 1)[..., None]
            fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
            out += (L[i, j] * (1 - fu) * (1 - fv) + L[i + 1, j] * fu * (1 - fv)
                    + L[i, j + 1] * (1 - fu) * fv + L[i + 1, j + 1] * fu * fv) * (1.0, 0.8, 0.7)[o]
        out = (out - out.mean()) * 1.8 + out.mean()
        return np.clip(out / 2.5 * 255, 0, 255)


def faces():
    """Axis-aligned textured rectangles: (axis, value, (a-axis, a0, a1), (b-axis, b0, b1), texture)."""
    W, D, H = ROOM
    x0, z0, x1, z1, top = TABLE
    F_ = [(1, 0.0, (0, 0, W), (2, 0, D)), (1, H, (0, 0, W), (2, 0, D)),
          (0, 0.0, (2, 0, D), (1, 0, H)), (0, W, (2, 0, D), (1, 0, H)),
          (2, 0.0, (0, 0, W), (1, 0, H)), (2, D, (0, 0, W), (1, 0, H)),
          (1, top, (0, x0, x1), (2, z0, z1)),
          (0, x0, (2, z0, z1), (1, 0, top)), (0, x1, (2, z0, z1), (1, 0, top)),
          (2, z0, (0, x0, x1), (1, 0, top)), (2, z1, (0, x0, x1), (1, 0, top))]
    return [(ax, val, a, b, ValueNoise(10 + k)) for k, (ax, val, a, b) in enumerate(F_)]


def render(R_cw, c, fcs):
    """Ray-cast the room: nearest face hit per pixel, textured in metres on that face."""
    v, u = np.mgrid[0:H_IMG, 0:W_IMG].astype(np.float64)
    d_cam = np.stack([(u + 0.5 - W_IMG / 2) / F, (v + 0.5 - H_IMG / 2) / F, np.ones_like(u)], -1)
    d = d_cam @ R_cw  # world ray directions (R_cw rows are the camera axes in world)
    best = np.full(u.shape, np.inf)
    img = np.zeros(u.shape + (3,))
    for ax, val, (aa, a0, a1), (bb, b0, b1), tex in fcs:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - c[ax]) / d[..., ax]
        pa = c[aa] + t * d[..., aa]
        pb = c[bb] + t * d[..., bb]
        hit = (t > 1e-3) & (t < best) & (pa >= a0) & (pa <= a1) & (pb >= b0) & (pb <= b1)
        if hit.any():
            best[hit] = t[hit]
            img[hit] = tex(pa[hit], pb[hit])
    return img.astype(np.uint8)


@pytest.mark.slow
def test_cpu_sfm_pipeline_recovers_room(tmp_path):
    fcs = faces()
    centers, rots = camera_loop(48, inward=True)
    frames = tmp_path / "shots"
    frames.mkdir()
    for i, (c, R) in enumerate(zip(centers, rots)):
        cv2.imwrite(str(frames / f"{i:04d}.jpg"), render(R, c, fcs), [cv2.IMWRITE_JPEG_QUALITY, 92])
    out = tmp_path / "scene"
    code = main(["run", str(frames), "--out", str(out), "--until", "occupancy", "--features", "sift", "--device", "cpu",
                 "--matcher", "exhaustive", "--focal-35mm", f"{F / 1200.0 * 43.27:.3f}", "--quiet"])
    errs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if '"error"' in l]
    assert code == 0, errs
    rep = json.loads((out / "report.json").read_text())
    assert rep["sfm"]["registeredRatio"] > 0.9, rep["sfm"]
    scene = json.loads((out / "scene.json").read_text())
    path = np.array(scene["cameraPath"])
    # cameras were 1.45 m up: the prior fixes scale, so the recovered loop must have the true size
    assert abs(np.median(path[:, 1]) - 1.45) < 0.05
    true_xz = centers[:, [0, 2]] - centers[:, [0, 2]].mean(0)
    got_xz = path[:, [0, 2]] - path[:, [0, 2]].mean(0)
    # the walked ellipse has semi-axes 1.6 x 0.9 m; compare its extent (invariant to the 90-deg yaw ambiguity)
    ext_true = sorted(np.ptp(true_xz, 0)); ext_got = sorted(np.ptp(got_xz, 0))
    assert np.allclose(ext_got, ext_true, rtol=0.06), (ext_got, ext_true)
    assert rep["align"]["floorTiltDeg"] < 25 and rep["occupancy"]["blocked"] > 0
