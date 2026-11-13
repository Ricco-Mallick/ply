"""plypipe.mushroom_fetch: the resumable download and the unpacking of a room, against a local server and small archives laid out like the
real ones (member names read from the honka and koivu archives and from Zenodo's file listings, 2026-11-13)."""
import hashlib
import io
import json
import re
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from test_mushroom import ROOM, dataset_json, random_similarity, walk, write_ply

from plypipe import geometry, mushroom, mushroom_fetch


class Zenodo(BaseHTTPRequestHandler):
    """Serves self.server.files by path, with Range support, and counts the bytes it sent."""

    def do_GET(self):
        body = self.server.files.get(self.path.lstrip("/"))
        if body is None:
            return self.send_error(404)
        lo, hi = 0, len(body) - 1
        if m := re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", "")):
            lo, hi = int(m[1]), min(int(m[2]), hi)
        self.send_response(206 if self.headers.get("Range") else 200)
        self.send_header("Content-Length", str(hi - lo + 1))
        self.end_headers()
        self.wfile.write(body[lo:hi + 1])
        self.server.sent += hi - lo + 1

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Zenodo)
    srv.files, srv.sent = {}, 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def tar_gz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.addfile(tarfile.TarInfo("room_datasets/honka/"))  # a directory entry, as in the real archives
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def entry(name, data, record=1):
    return {"record": record, "name": name, "bytes": len(data), "md5": hashlib.md5(data).hexdigest()}


def test_download_resumes_skips_and_refuses_a_file_that_differs_from_the_record(tmp_path, server, monkeypatch):
    monkeypatch.setattr(mushroom_fetch, "CHUNK", 1000)
    data = np.random.default_rng(0).integers(0, 256, 10_500, dtype=np.uint8).tobytes()
    server.files["a/blob"] = data
    e, url, dest = entry("blob", data), f"http://127.0.0.1:{server.server_port}/a/blob", tmp_path / "blob"
    part = tmp_path / "blob.part"
    part.write_bytes(b"\0" * len(data))  # an earlier run got ranges 0 to 3 and 5
    part.write_bytes(data[:4000] + b"\0" * 1000 + data[5000:6000] + b"\0" * (len(data) - 6000))
    (tmp_path / "blob.chunks").write_text("0\n1\n2\n3\n5\n")
    assert mushroom_fetch.download(e, dest, url, streams=3, log=lambda *a: None) == "downloaded"
    assert dest.read_bytes() == data and not part.exists() and server.sent == len(data) - 5000  # only the six missing ranges came over
    assert mushroom_fetch.download(e, dest, url, log=lambda *a: None) == "present" and server.sent == len(data) - 5000
    bad = tmp_path / "bad"
    with pytest.raises(ValueError, match="md5 differs"):
        mushroom_fetch.download({**e, "md5": "0" * 32}, bad, url, log=lambda *a: None)
    assert not bad.exists() and not (tmp_path / "bad.part").exists()


def make_room(rng):
    """The honka archives in miniature: 12 photos (frame_00005 has no pose), poses in a frame the alignment moves to the scan's."""
    c2w = walk(11)
    s, R, t = random_similarity(rng, 1.0)
    to_scan = np.eye(4)
    to_scan[:3, :3], to_scan[:3, 3] = R, t
    stored = np.linalg.inv(to_scan)[None] @ c2w
    nums = [i + 1 if i < 4 else i + 2 for i in range(11)]  # 1..4, 6..12: frame 5 is unposed
    meta = dataset_json(stored, gaps=False)
    for f, k in zip(meta["frames"], nums):
        f["file_path"] = f"./images/frame_{k:05d}.jpg"
    colmap = {**{k: v for k, v in meta.items() if k != "frames"}, "frames": [json.loads(json.dumps(f)) for f in meta["frames"][1:]]}  # lost a frame
    for f in colmap["frames"]:
        f["transform_matrix"][0][3] += 30.0  # and sits far from the scan, so the other file is the one the reference takes
    pc = rng.uniform(0, 1, (6000, 3)) * ROOM
    return c2w, to_scan, meta, colmap, pc, nums


def test_a_room_is_downloaded_unpacked_and_its_reference_built_in_the_scan_frame(tmp_path, server, monkeypatch):
    rng = np.random.default_rng(1)
    c2w, to_scan, meta, colmap, scan, nums = make_room(rng)
    long = "room_datasets/honka/iphone/long_capture/"
    ply = tmp_path / "gt_pd.ply"
    write_ply(ply, scan)
    mesh = tmp_path / "gt_mesh.ply"
    write_ply(mesh, scan[:50], faces=[(0, 1, 2)])
    iphone = tar_gz({**{long + f"images/frame_{k:05d}.jpg": b"jpg%d" % k for k in range(1, 13)},
                     **{long + f"depth/frame_{k:05d}.png": b"png" for k in range(1, 13)},
                     long + "transformations.json": json.dumps(meta).encode(), long + "transformations_colmap.json": json.dumps(colmap).encode(),
                     long + "test.txt": b"frame_00011\n", long + "polycam_pointcloud.ply": b"polycam",
                     "room_datasets/honka/iphone/short_capture/images/frame_00001.jpg": b"short"})
    scan_tar = tar_gz({"room_datasets/honka/gt_mesh.ply": mesh.read_bytes(), "room_datasets/honka/gt_pd.ply": ply.read_bytes(),
                       "room_datasets/honka/icp_iphone.json": json.dumps({"gt_transformation": np.linalg.inv(to_scan).ravel().tolist()}).encode()})
    server.files = {"1/honka_iphone.tar.gz": iphone, "2/honka_mesh_pd.tar.gz": scan_tar}
    monkeypatch.setattr(mushroom, "ZENODO", f"http://127.0.0.1:{server.server_port}/{{record}}/{{name}}")
    monkeypatch.setitem(mushroom.ROOMS, "honka", {"iphone": entry("honka_iphone.tar.gz", iphone, 1), "scan": entry("honka_mesh_pd.tar.gz", scan_tar, 2)})
    real = mushroom.build_reference
    monkeypatch.setattr(mushroom, "build_reference", lambda poses, icp, pts, names: real(poses, icp, pts, names, cell=0.1))  # the miniature room is 6 m wide

    root = tmp_path / "mushroom"
    summary = mushroom_fetch.seed_room("honka", root, tmp_path / "scratch", log=lambda *a: None)
    assert summary["downloads"] == {"iphone": "downloaded", "scan": "downloaded"} and summary["photos"] == 12 and summary["poseFrames"] == 11
    assert sorted(p.name for p in (root / "honka" / "in").iterdir()) == [f"frame_{k:05d}.jpg" for k in range(1, 13)]  # no depth, no short capture
    ref = mushroom.load_reference(root / "honka" / "reference")
    assert ref["meta"]["poses"] == "transformations.json" and ref["meta"]["direction"]["direction"] == "inverted" and ref["meta"]["direction"]["confident"]
    assert ref["meta"]["scanFile"] == "gt_pd.ply" and ref["meta"]["licence"] == "CC-BY-4.0" and len(ref["inputNames"]) == 12
    assert np.allclose(ref["centres"], c2w[:, :3, 3], atol=1e-3) and 0 < len(ref["points"]) < len(scan)
    assert (root / "honka" / "reference" / "icp_iphone.json").exists() and not (tmp_path / "scratch").exists()
    # twin frames 000000 ... are the 12 photos in name order: frame_00005 is the twin's 000004 and has no reference pose
    assert mushroom.match(["000003.jpg", "000004.jpg", "000005.jpg"], ref["names"], ref["inputNames"])[1].tolist() == [3, 4]
    before = server.sent
    assert "already on the Volume" in mushroom_fetch.seed_room("honka", root, tmp_path / "scratch", log=lambda *a: None)["skipped"]
    (root / "honka" / "reference" / "meta.json").unlink()                      # a rebuild takes the archives that are there, and fetches nothing
    again = mushroom_fetch.seed_room("honka", root, tmp_path / "scratch", log=lambda *a: None)
    assert again["downloads"] == {"iphone": "present", "scan": "present"} and server.sent == before


def test_the_laser_meshs_vertices_stand_in_when_the_archive_has_no_cloud(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    scan = rng.uniform(0, 1, (3000, 3)) * ROOM
    mesh = tmp_path / "gt_mesh.ply"
    write_ply(mesh, scan, faces=[(0, 1, 2)])
    archive = tmp_path / "honka_mesh_pd.tar.gz"
    archive.write_bytes(tar_gz({"room_datasets/honka/gt_mesh.ply": mesh.read_bytes(), "room_datasets/honka/icp_iphone.json": b"{}"}))
    iphone = tmp_path / "honka_iphone.tar.gz"
    iphone.write_bytes(tar_gz({"room_datasets/honka/iphone/long_capture/images/frame_00001.jpg": b"j"}))
    monkeypatch.setitem(mushroom.ROOMS, "honka", {"iphone": {"name": iphone.name}, "scan": {"name": archive.name}})
    logs = []
    found = mushroom_fetch.unpack_room("honka", tmp_path, tmp_path / "out", tmp_path / "scratch", log=logs.append)
    assert "gt_mesh.ply" in found["raw"] and "gt_pd.ply" not in found["raw"] and "stand in" in logs[0]
    assert np.array_equal(mushroom.read_ply_points(tmp_path / "scratch" / "raw" / "gt_mesh.ply"), scan.astype(np.float32))


def test_the_estimate_counts_both_archives_at_the_measured_rate_and_the_stop_loss_is_one_and_a_half_times_it():
    e = mushroom_fetch.fetch_estimate("honka")
    assert e["bytes"] == 170_700_232 + 239_649_902 and e["seconds"] == round(e["bytes"] / 1.85e6 + 90)
    assert e["timeoutSeconds"] >= 1.5 * e["seconds"] and e["timeoutSeconds"] % 60 == 0
    assert mushroom_fetch.fetch_estimate("coffee_room")["bytes"] == 183_080_957 + 288_148_502


def test_registered_archives_are_the_ones_zenodo_lists():
    for room, spec in mushroom.ROOMS.items():
        assert spec["iphone"]["name"] == f"{room}_iphone.tar.gz" and spec["scan"]["name"] == f"{room}_mesh_pd.tar.gz"
        assert mushroom.archive_url(spec["scan"]) == f"https://zenodo.org/api/records/10222321/files/{room}_mesh_pd.tar.gz/content"
        assert re.fullmatch(r"[0-9a-f]{32}", spec["iphone"]["md5"]) and re.fullmatch(r"[0-9a-f]{32}", spec["scan"]["md5"])
