"""AR scans end to end on CPU: synthetic room renders packed exactly like the capture app's AR tar (keyframe JPEGs +
poses.json with OpenGL camera-to-world matrices in a 'local-floor' space), through frames -> sfm -> align. The phone
poses must fix the scale even when the camera-height prior is wrong, and leave the floor at y = 0."""
import io
import json
import tarfile

import cv2
import numpy as np
import pytest

pytest.importorskip("pycolmap")
from plypipe import geometry  # noqa: E402
from plypipe.cli import main  # noqa: E402
from synth import camera_loop  # noqa: E402
from test_sfm_e2e import F, H_IMG, W_IMG, faces, render  # noqa: E402


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(30, 3))
    R = geometry.rot_y(0.7) @ geometry.rotation_between(np.array([0, 1.0, 0]), np.array([0.2, 1, 0.1]))
    b = (2.7 * (R @ a.T)).T + [1, -2, 0.5]
    s, R2, t = geometry.umeyama(a, b)
    assert abs(s - 2.7) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [1, -2, 0.5])


def ar_tar(path, centers, rots, fcs, yaw=0.7, shift=(0.3, 0.0, -1.2)):
    """The app's tar: frames/NNNNN.jpg + poses.json. ARCore's origin and heading are arbitrary, gravity and floor are not."""
    Ry = geometry.rot_y(yaw)
    kfs, buf = [], io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i, (c, R_cw) in enumerate(zip(centers, rots)):
            ok, jpg = cv2.imencode(".jpg", render(R_cw, c, fcs), [cv2.IMWRITE_JPEG_QUALITY, 92])
            name = f"frames/{i:05d}.jpg"
            info = tarfile.TarInfo(name)
            info.size = len(jpg)
            tf.addfile(info, io.BytesIO(jpg.tobytes()))
            # OpenGL camera axes in world: x right, y up (= -down), z back (= -forward)
            M = np.eye(4)
            M[:3, 0], M[:3, 1], M[:3, 2] = Ry @ R_cw[0], Ry @ -R_cw[1], Ry @ -R_cw[2]
            M[:3, 3] = Ry @ c + shift
            kfs.append({"file": name, "t": i * 300, "m": M.T.reshape(-1).tolist(), "fx": F, "fy": F, "cx": W_IMG / 2, "cy": H_IMG / 2, "w": W_IMG, "h": H_IMG})
        meta = json.dumps({"version": 1, "kind": "ply-ar", "keyframes": kfs}).encode()
        info = tarfile.TarInfo("poses.json")
        info.size = len(meta)
        tf.addfile(info, io.BytesIO(meta))
    path.write_bytes(buf.getvalue())


def test_ar_scan_gives_scale_and_level_floor(tmp_path):
    fcs = faces()
    centers, rots = camera_loop(48, inward=True)
    centers[:, 1] += 0.18 * np.sin(np.linspace(0, 4 * np.pi, len(centers)))  # hand height wanders
    inp = tmp_path / "input"
    inp.mkdir()
    ar_tar(inp / "scan-01.tar", centers, rots, fcs)
    out = tmp_path / "scene"
    # a deliberately wrong camera height: with AR poses it must not matter
    code = main(["run", str(inp), "--out", str(out), "--until", "align", "--features", "sift", "--device", "cpu",
                 "--matcher", "exhaustive", "--height-cam", "3.0", "--quiet"])
    errs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if '"error"' in l]
    assert code == 0, errs
    rep = json.loads((out / "report.json").read_text())
    assert rep["frames"]["source"] == "ar" and rep["frames"]["kept"] >= 44
    assert rep["align"]["scaleSource"] == "ar-poses", rep["align"]
    assert rep["align"]["ar"]["residualMedianM"] < 0.05, rep["align"]
    path = np.array(json.loads((out / "scene.json").read_text())["cameraPath"])
    assert abs(np.median(path[:, 1]) - np.median(centers[:, 1])) < 0.04  # true heights, floor at 0
    ext_true = sorted(np.ptp(centers[:, [0, 2]], 0)); ext_got = sorted(np.ptp(path[:, [0, 2]], 0))
    assert np.allclose(ext_got, ext_true, rtol=0.04), (ext_got, ext_true)
    # the coverage preview the app shows: aligned sparse points + every frame mapped back to its scan and second
    from plypipe import preview
    preview.build(out)
    meta = json.loads((out / "preview.json").read_text())
    blob = (out / "preview.bin").read_bytes()
    n = int(np.frombuffer(blob[:4], np.uint32)[0])
    assert meta["aligned"] and meta["registered"] == meta["total"] >= 44 and n == meta["points"] > 200
    assert len(blob) == 4 + n * 16 and all(f["clip"] == "scan-01.tar" for f in meta["frames"])
    ts = [f["t"] for f in meta["frames"]]
    assert ts == sorted(ts) and ts[-1] > 12  # 48 keyframes 0.3 s apart, a few soft ones dropped
