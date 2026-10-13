"""Queue a rehearsal job: a synthetic room rendered to a short video, uploaded through the real Ply API like the app does.

    python pipeline/kaggle/rehearsal.py cpu   # for a CPU-only Kaggle worker: everything up to a training dry run
    python pipeline/kaggle/rehearsal.py gpu   # GPU smoke test: the whole pipeline with 300 training steps (~5 min)

Capture ids start with "rehearsal-", so a rehearsal worker (PLY_CLAIM_PREFIX) never takes a real capture.
Reads APP_KEY from cloud/.secrets. Prints the job id.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "tests")]
API = "https://ply-api.ricco-ply.workers.dev"
UA = "ply-rehearsal/1.0 (+https://ricco.codes/ply)"


def secrets() -> dict:
    out = {}
    for line in (HERE.parent / "cloud" / ".secrets").read_text().splitlines():
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def call(method: str, path: str, key: str, body: bytes | dict | None = None, ctype: str = "application/json"):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(API + path, data=data, method=method, headers={"x-ply-key": key, "content-type": ctype, "user-agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def render_video(dest: Path) -> float:
    """48 ray-traced views of the synthetic room (the SfM e2e test's scene) as a 12 s, 4 fps MP4. Returns focal in 35 mm."""
    import cv2
    from synth import camera_loop
    from test_sfm_e2e import F, faces, render
    fcs = faces()
    centers, rots = camera_loop(48, inward=True)
    with tempfile.TemporaryDirectory() as td:
        for i, (c, R) in enumerate(zip(centers, rots)):
            cv2.imwrite(f"{td}/{i:04d}.png", render(R, c, fcs))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4", "-i", f"{td}/%04d.png",
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(dest)], check=True)
    return F / 1200.0 * 43.27  # same 35 mm-equivalent focal the SfM test uses


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    key = secrets()["APP_KEY"]
    cap = f"rehearsal-{mode}-{int(time.time())}"
    with tempfile.TemporaryDirectory() as td:
        vid = Path(td) / "clip-01.mp4"
        focal35 = render_video(vid)
        blob = vid.read_bytes()
        up = call("POST", "/v1/uploads", key, {"captureId": cap, "name": "clip-01.mp4", "type": "video/mp4"})
        url = call("POST", "/v1/uploads/sign", key, {"key": up["key"], "uploadId": up["uploadId"], "parts": [1]})["urls"]["1"]
        with urllib.request.urlopen(urllib.request.Request(url, data=blob, method="PUT", headers={"user-agent": UA}), timeout=120) as r:
            etag = r.headers["ETag"]
        call("POST", "/v1/uploads/complete", key, {"key": up["key"], "uploadId": up["uploadId"], "parts": [{"partNumber": 1, "etag": etag}]})
    cj = f"captures/{cap}/capture.json"
    call("PUT", "/v1/files?key=" + cj, key, json.dumps({"version": 1, "name": cap, "rehearsal": mode}).encode())
    common = ["--features", "sift", "--matcher", "exhaustive", "--fps", "4", "--focal-35mm", f"{focal35:.3f}"]
    args = (["--preset", "quick", *common, "--device", "cpu", "--train-dry-run", "--until", "train"] if mode == "cpu"
            else ["--preset", "quick", *common, "--iterations", "300"])
    job = call("POST", "/v1/jobs", key, {"captureId": cap, "name": f"Rehearsal ({mode})", "settings": {"preset": "quick", "rehearsal": mode},
                                         "args": args, "files": [up["key"], cj]})
    print(json.dumps({"jobId": job["jobId"], "capture": cap, "videoBytes": len(blob), "args": args}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
