"""Live view of a running job: what each stage has produced so far, published next to the results for the status page.

    live-index.json        what exists, with timestamps (the status page polls this; served uncached)
    live-frames.jpg        contact sheet of the kept frames (up to 160 thumbnails) + live-frames.json (grid layout)
    live-sparse.bin/.json  the camera-solving result (preview.py format): after sfm, then again aligned after align
    live-train-<step>.spz  the twin during training (SH0, decimated to ~250k Gaussians: a few MB each)
    loss curve             in live-index.json (training step fraction, loss)

Everything runs on background threads so the pipeline never waits for an upload, and any failure here is logged and
ignored: the live view is a nicety, the job must not fail because of it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

SNAP_MAX = 250_000
LOSS = re.compile(r"loss=([0-9.]+)")
SPLAT_TRANSFORM = "@playcanvas/splat-transform@3.10.1"


class Live:
    def __init__(self, api, jid: str, out: Path, log=print):
        self.api, self.jid, self.out, self.log = api, jid, out, log
        self.index = {"v": 1, "job": jid, "updated": time.time(), "stages": {}, "snapshots": [], "loss": []}
        self.lock = threading.Lock()
        self.seen_snaps: set[str] = set()
        self.last_loss = -1.0
        self.sparse_lock = threading.Lock()  # sfm and align previews must not interleave
        self.preview_only = False  # coverage checks publish their own preview; the live sparse step is skipped

    # ---- plumbing ----
    def _put(self, name: str, path: Path, ctype: str):
        self.api.upload(f"scenes/{self.jid}/{name}", path, ctype)

    def _publish(self):
        with self.lock:  # write and upload under the lock: another thread may be about to rewrite it
            self.index["updated"] = time.time()
            f, tmp = self.out / "live-index.json", self.out / "live-index.json.tmp"
            tmp.write_text(json.dumps(self.index))
            os.replace(tmp, f)
            self._put("live-index.json", f, "application/json")

    def _bg(self, fn, *a):
        def go():
            try:
                fn(*a)
            except Exception as e:  # never fail the job over the live view
                self.log(f"[live] {fn.__name__} failed: {e}")
        threading.Thread(target=go, daemon=True).start()

    def stage(self, name: str, status: str, **info):
        with self.lock:
            self.index["stages"][name] = {"status": status, "t": time.time(), **info}
        self._bg(self._publish)

    # ---- stage outputs ----
    def stage_done(self, name: str):
        self.stage(name, "done")
        if name == "frames":
            self._bg(self.frames_sheet)
        elif name in ("sfm", "align") and not self.preview_only:
            self._bg(self.sparse, name)

    def frames_sheet(self):
        import cv2
        files = sorted((self.out / "frames").glob("*.jpg"))
        if not files:
            return
        pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(160, len(files)))]
        w = 128
        thumbs = []
        for p in pick:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h = round(img.shape[0] * w / img.shape[1])
            thumbs.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
        if not thumbs:
            return
        th = max(t.shape[0] for t in thumbs)
        cols = 16
        rows = -(-len(thumbs) // cols)
        sheet = np.zeros((rows * th, cols * w, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            sheet[r * th:r * th + t.shape[0], c * w:(c + 1) * w] = t
        f = self.out / "live-frames.jpg"
        cv2.imwrite(str(f), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._put("live-frames.jpg", f, "image/jpeg")
        with self.lock:
            self.index["frames"] = {"file": "live-frames.jpg", "count": len(files), "shown": len(thumbs), "cols": cols, "w": w, "h": th, "t": time.time()}
        self._publish()

    def sparse(self, stage: str):
        from plypipe import preview
        with self.sparse_lock:
            self._sparse(stage, preview)

    def _sparse(self, stage: str, preview):
        bin_, js = preview.build(self.out, self.out / f"_live_{stage}")
        for src, name, ct in ((bin_, "live-sparse.bin", "application/octet-stream"), (js, "live-sparse.json", "application/json")):
            dst = self.out / name
            dst.write_bytes(src.read_bytes())
            self._put(name, dst, ct)
        meta = json.loads(js.read_text())
        with self.lock:
            self.index["sparse"] = {"file": "live-sparse.bin", "meta": "live-sparse.json", "after": stage, "aligned": meta["aligned"],
                                    "points": meta["points"], "registered": meta["registered"], "total": meta["total"], "t": time.time()}
        self._publish()

    # ---- training ----
    def train_line(self, frac: float | None, msg: str):
        """A training progress line: keep a sampled loss curve (every 1 % of training)."""
        m = LOSS.search(msg or "")
        if not m or frac is None or frac - self.last_loss < 0.01:
            return
        self.last_loss = frac
        with self.lock:
            self.index["loss"].append([round(frac, 3), float(m.group(1))])

    def poll_snapshots(self, alive, final_step: int | None = None):
        """While training runs, turn each new intermediate PLY into a small SPZ and publish it (`alive` turns false when
        training ends; the final PLY is skipped: export publishes the finished twin)."""
        d = self.out / "train" / "ply"
        last_pub = 0.0
        while alive():
            for p in sorted(d.glob("point_cloud_*.ply")) if d.exists() else []:
                if p.name in self.seen_snaps or time.time() - p.stat().st_mtime < 5:  # still being written
                    continue
                if final_step is not None and int(p.stem.rsplit("_", 1)[1]) >= final_step - 1:
                    self.seen_snaps.add(p.name)
                    continue
                self.seen_snaps.add(p.name)
                try:
                    self.snapshot(p)
                except Exception as e:
                    self.log(f"[live] snapshot {p.name} failed: {e}")
            if time.time() - last_pub > 60 and self.index["loss"]:  # the loss curve, once a minute
                last_pub = time.time()
                try:
                    self._publish()
                except Exception as e:
                    self.log(f"[live] publish failed: {e}")
            time.sleep(15)

    def snapshot(self, ply: Path):
        step = int(ply.stem.rsplit("_", 1)[1])
        small, spz = self.out / f"live-snap-{step}.ply", self.out / f"live-train-{step}.spz"
        st = ["npx", "-y", SPLAT_TRANSFORM, "-q", "-w"]
        subprocess.run([*st, str(ply), "-N", "-H", "0", "-d", str(SNAP_MAX), str(small)], check=True, capture_output=True, timeout=600)
        subprocess.run([*st, "--spz-version", "3", str(small), str(spz)], check=True, capture_output=True, timeout=600)
        small.unlink(missing_ok=True)
        self._put(spz.name, spz, "application/octet-stream")
        with self.lock:
            self.index["snapshots"].append({"step": step, "file": spz.name, "mb": round(spz.stat().st_size / 2**20, 2), "t": time.time()})
        self._publish()
