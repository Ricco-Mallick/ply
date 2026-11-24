"""iPhone LiDAR captures (Stray Scanner exports) end to end on CPU: a synthetic room rendered into rgb.mp4 with an
odometry.csv in ARKit's convention (metres, +y up, arbitrary start and heading), zipped the way an iPhone's Files app
compresses a folder, through frames -> sfm -> align. The ARKit positions must give the scale and a level floor, and the
kept frames' depth maps must land in lidar/."""
import json
import shutil
import subprocess
import zipfile

import cv2
import numpy as np
import pytest

pytest.importorskip("pycolmap")
from plypipe import geometry, stray  # noqa: E402
from plypipe.cli import main  # noqa: E402
from synth import camera_loop  # noqa: E402
from test_sfm_e2e import F, H_IMG, W_IMG, faces, render  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def stray_scan(d, centers, rots, fcs, yaw=-1.1, shift=(0.4, 1.2, -0.7), fps=2, per_frame_k=True):
    """A Stray Scanner folder: rgb.mp4, odometry.csv, camera_matrix.csv, depth/ and confidence/."""
    (d / "depth").mkdir(parents=True)
    (d / "confidence").mkdir()
    frames = d / "_png"
    frames.mkdir()
    Ry = geometry.rot_y(yaw)
    head = "timestamp, frame, x, y, z, qx, qy, qz, qw" + (", fx, fy, cx, cy" if per_frame_k else "")
    lines = [head]
    for i, (c, R_cw) in enumerate(zip(centers, rots)):
        cv2.imwrite(str(frames / f"{i:06d}.png"), render(R_cw, c, fcs))
        p = Ry @ c + shift  # ARKit's origin is where the session started, its heading arbitrary; gravity is +y
        k = f", {F}, {F}, {W_IMG / 2}, {H_IMG / 2}" if per_frame_k else ""
        lines.append(f"{100.0 + i / fps:.4f}, {i:06d}, {p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f}, 0, 0, 0, 1{k}")
        cv2.imwrite(str(d / "depth" / f"{i:06d}.png"), np.full((192, 256), 2000 + i, np.uint16))
        cv2.imwrite(str(d / "confidence" / f"{i:06d}.png"), np.full((192, 256), 2, np.uint8))
    (d / "odometry.csv").write_text("\n".join(lines) + "\n")
    np.savetxt(d / "camera_matrix.csv", np.array([[F, 0, W_IMG / 2], [0, F, H_IMG / 2], [0, 0, 1]]), delimiter=",")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", str(d / "rgb.mp4")], check=True)
    shutil.rmtree(frames)


def zip_folder(folder, zp):
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            z.write(p, f"{folder.name}/{p.relative_to(folder)}")
        z.writestr("__MACOSX/._junk", b"x")  # what macOS / iOS add: must be ignored


def test_odometry_reader_tolerates_spaces_and_old_exports(tmp_path):
    d = tmp_path / "scan"
    d.mkdir()
    (d / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n1.0, 000000, 0, 1.4, 0, 0, 0, 0, 1\n"
                                    "1.5, 000001, 0.5, 1.4, 0, 0, 0.7071068, 0, 0.7071068\n")
    rows = stray.read_odometry(d)
    assert [r["frame"] for r in rows] == [0, 1] and rows[1]["p"] == [0.5, 1.4, 0.0] and rows[0]["K"] == [None] * 4
    m = np.array(stray.pose_m(rows[1])).reshape(4, 4).T  # column-major, translation at 12:15
    assert np.allclose(m[:3, 3], [0.5, 1.4, 0]) and np.allclose(m[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)


def test_unzip_refuses_paths_outside(tmp_path):
    zp = tmp_path / "bad.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        stray.unzip(zp, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


@needs_ffmpeg
@pytest.mark.parametrize("per_frame_k", [True, pytest.param(False, marks=pytest.mark.slow)])  # old exports: camera_matrix.csv only
def test_iphone_lidar_zip_gives_scale_level_floor_and_depth(tmp_path, per_frame_k):
    fcs = faces()
    centers, rots = camera_loop(48, inward=True)
    centers[:, 1] += 0.15 * np.sin(np.linspace(0, 4 * np.pi, len(centers)))
    scan = tmp_path / "raw" / "7a3f9c01"
    stray_scan(scan, centers, rots, fcs, per_frame_k=per_frame_k)
    inp = tmp_path / "input"
    inp.mkdir()
    zip_folder(scan, inp / "7a3f9c01.zip")
    out = tmp_path / "scene"
    code = main(["run", str(inp), "--out", str(out), "--until", "align", "--features", "sift", "--device", "cpu",
                 "--matcher", "exhaustive", "--height-cam", "3.0", "--quiet"])
    errs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if '"error"' in l]
    assert code == 0, errs
    rep = json.loads((out / "report.json").read_text())
    assert rep["frames"]["source"] == "iphone-lidar" and rep["frames"]["kept"] >= 44, rep["frames"]
    assert rep["frames"]["depthMaps"] == rep["frames"]["kept"]
    assert rep["align"]["scaleSource"] == "ar-poses", rep["align"]
    assert rep["align"]["ar"]["residualMedianM"] < 0.05, rep["align"]
    path = np.array(json.loads((out / "scene.json").read_text())["cameraPath"])
    ext_true = sorted(np.ptp(centers[:, [0, 2]], 0)); ext_got = sorted(np.ptp(path[:, [0, 2]], 0))
    assert np.allclose(ext_got, ext_true, rtol=0.04), (ext_got, ext_true)  # true metres despite --height-cam 3.0
    assert abs(np.median(path[:, 1]) - np.median(centers[:, 1])) < 0.04  # floor at y = 0
    lid = json.loads((out / "lidar.json").read_text())
    some = next(iter(lid["frames"].values()))
    d = cv2.imread(str(out / "lidar" / some["depth"]), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and d.shape == (192, 256) and 2000 <= int(d[0, 0]) < 2048
    assert not (out / "_lidar_in").exists() and not (out / "_lidar_tars").exists()
