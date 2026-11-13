# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The I/O half of pipeline/modal/mushroom_seed.py, kept free of Modal so a CPU test can run it against a local server and small archives:
a resumable parallel download with an md5 check, and the unpacking of one room's two archives into the layout ply_eval's scenarios read.

Layout under <root> (the Volume's mushroom/ folder):
    _dl/<archive>                 the downloaded archives, kept: a rerun skips them (the md5 was checked before the file got its name)
    <room>/in/frame_NNNNN.jpg     the long capture's photos, what `plypipe run` is given
    <room>/reference/             meta.json, ref_points.npy (the laser cloud, thinned, in the scan frame), the pose files, icp_iphone.json
    <room>/job/                   written later, by the solve step
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from plypipe import mushroom

CHUNK = 16 << 20
STREAMS = 4
# Measured from the development sandbox on 2026-11-13: one stream of Zenodo's honka_mesh_pd.tar.gz ran at 1.85 MB/s (32 MB in 17.3 s).
# The estimate counts one stream's rate and ignores the parallel streams' gain, so it is the slow side.
BYTES_PER_SECOND = 1.85e6
UNPACK_SECONDS = 90  # streaming 240 to 290 MB of gzip, reading the laser cloud and thinning it, writing 330 to 390 photos
STOP_LOSS = 1.5      # IMPROVEMENT-PLAN rule 3


def fetch_estimate(room: str) -> dict:
    r = mushroom.ROOMS[room]
    size = r["iphone"]["bytes"] + r["scan"]["bytes"]
    seconds = size / BYTES_PER_SECOND + UNPACK_SECONDS
    return {"room": room, "bytes": size, "seconds": round(seconds), "timeoutSeconds": math.ceil(STOP_LOSS * seconds / 60) * 60}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(entry: dict, dest: Path, url: str | None = None, streams: int = STREAMS, log=print) -> str:
    """Fetch an archive in 16 MB ranges over `streams` connections into <dest>.part, remember finished ranges in <dest>.chunks (a killed run
    resumes there), check size and md5, and only then give the file its name. Returns "present" when the verified file was already there."""
    url, size = url or mushroom.archive_url(entry), entry["bytes"]
    if dest.exists() and dest.stat().st_size == size:
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part, marks = Path(f"{dest}.part"), Path(f"{dest}.chunks")
    count = math.ceil(size / CHUNK)
    done = {int(x) for x in marks.read_text().split()} if part.exists() and part.stat().st_size == size and marks.exists() else set()
    if not done:
        with part.open("wb") as f:
            f.truncate(size)
        marks.write_text("")
    lock, fd, started = threading.Lock(), os.open(part, os.O_WRONLY), time.time()

    def work(i: int) -> None:
        lo, hi = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"}), timeout=120) as r:
                    data = r.read()
                if len(data) != hi - lo + 1:
                    raise OSError(f"range {lo}-{hi}: got {len(data)} bytes")
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        os.pwrite(fd, data, lo)
        with lock:
            done.add(i)
            with marks.open("a") as m:
                m.write(f"{i}\n")

    try:
        with ThreadPoolExecutor(streams) as pool:
            list(pool.map(work, [i for i in range(count) if i not in done]))
    finally:
        os.close(fd)
    if part.stat().st_size != size or md5_of(part) != entry["md5"]:
        part.unlink()
        marks.unlink(missing_ok=True)
        raise ValueError(f"{entry['name']}: size or md5 differs from Zenodo's record; the download was discarded")
    part.rename(dest)
    marks.unlink(missing_ok=True)
    log(f"[mushroom] {entry['name']}: {size / 2**20:.0f} MB in {time.time() - started:.0f} s")
    return "downloaded"


def _walk(archive: Path, room: str):
    """(routing, name, file object) for each member of a .tar.gz that the seed wants, read as a stream."""
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            target = mushroom.member_target(room, m.name) if m.isfile() else None
            if target:
                yield (*target, tf.extractfile(m))


def _copy(src, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(src, f, 8 << 20)


def unpack_room(room: str, downloads: Path, root: Path, scratch: Path, log=print) -> dict:
    """Both archives -> <root>/<room>/in (photos) and <scratch>/raw (pose files, alignment, laser cloud; the mesh only if the cloud is
    missing, since the cloud is what the README names for geometry comparison). Returns what was found."""
    out, raw = root / room / "in", scratch / "raw"
    found = {"photos": 0, "raw": []}
    for routing, name, f in _walk(downloads / mushroom.ROOMS[room]["iphone"]["name"], room):
        if routing == "in":
            _copy(f, out / name)
            found["photos"] += 1
        else:
            _copy(f, raw / name)
            found["raw"].append(name)
    scan = downloads / mushroom.ROOMS[room]["scan"]["name"]
    for routing, name, f in _walk(scan, room):
        if name != "gt_mesh.ply":
            _copy(f, raw / name)
            found["raw"].append(name)
    if "gt_pd.ply" not in found["raw"]:
        log(f"[mushroom] {room}: no gt_pd.ply in {scan.name}; the laser mesh's vertices stand in for the cloud")
        for routing, name, f in _walk(scan, room):
            if name == "gt_mesh.ply":
                _copy(f, raw / name)
                found["raw"].append(name)
    return found


def build_room_reference(room: str, root: Path, scratch: Path, log=print) -> dict:
    """The scan-frame reference of an unpacked room: <root>/<room>/reference/{meta.json, ref_points.npy, the pose files, icp_iphone.json}."""
    import numpy as np
    raw, ref = scratch / "raw", root / room / "reference"
    cloud_file = raw / "gt_pd.ply" if (raw / "gt_pd.ply").exists() else raw / "gt_mesh.ply"
    scan = mushroom.read_ply_points(cloud_file, limit=20_000_000)
    log(f"[mushroom] {room}: {len(scan):,} points read from {cloud_file.name}")
    poses = {n: raw / n for n in ("transformations.json", "transformations_colmap.json") if (raw / n).exists()}
    meta, cloud = mushroom.build_reference(poses, raw / "icp_iphone.json", scan, [p.stem for p in (root / room / "in").glob("*.jpg")])
    ref.mkdir(parents=True, exist_ok=True)
    for p in (*poses.values(), raw / "icp_iphone.json", *([raw / "test.txt"] if (raw / "test.txt").exists() else [])):
        shutil.copy2(p, ref / p.name)
    np.save(ref / "ref_points.npy", cloud.astype(np.float32))
    spec = mushroom.ROOMS[room]
    meta.update({"room": room, "licence": "CC-BY-4.0", "source": {k: {x: spec[k][x] for x in ("record", "name", "bytes", "md5")} for k in ("iphone", "scan")},
                 "scanFile": cloud_file.name, "scanPointsRead": int(len(scan)), "photos": len(meta["inputNames"]), "poseFrames": len(meta["names"])})
    (ref / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def seed_room(room: str, root: Path, scratch: Path, log=print) -> dict:
    """Download (skipping what is there), unpack and build one room's input and reference under <root>. Returns a summary for the log."""
    started, downloads = time.time(), root / "_dl"
    got = {k: download(mushroom.ROOMS[room][k], downloads / mushroom.ROOMS[room][k]["name"], log=log) for k in ("iphone", "scan")}
    unpacked = unpack_room(room, downloads, root, scratch, log)
    meta = build_room_reference(room, root, scratch, log)
    shutil.rmtree(scratch, ignore_errors=True)
    return {"room": room, "downloads": got, "photos": unpacked["photos"], "poseFrames": meta["poseFrames"], "direction": meta["direction"],
            "candidates": meta["candidates"], "seconds": round(time.time() - started)}
