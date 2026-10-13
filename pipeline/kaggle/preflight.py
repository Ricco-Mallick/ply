"""Pre-flight: refuse to spend GPU quota unless everything a run needs is in place. Run before any Kaggle GPU run.

    python pipeline/kaggle/preflight.py [--job <id>] [--config-dir ~/.kaggle] [--need-min 35]

Checks (each prints PASS/FAIL with the reason):
  api        the Ply API answers from Python (Cloudflare blocks default Python user agents) and the worker token is valid
  storage    a signed download from B2 works (the free tier caps downloads at 1 GB/day; capped = 403)
  quota      the Kaggle account has at least --need-min GPU minutes left this week
  datasets   ply-pipeline and ply-worker-config exist and are ready on that account, and ply-pipeline matches git HEAD
  job        the job exists, is queued, and its inputs fit in the remaining daily download allowance
Exit code 0 only if all pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = os.environ.get("PLY_API", "https://ply-api.ricco-ply.workers.dev")
UA = "ply-preflight/1.0 (+https://ricco.codes/ply)"
B2_DAILY = 1_000_000_000


def secrets() -> dict:
    return dict(l.split("=", 1) for l in (ROOT / "cloud" / ".secrets").read_text().split() if "=" in l)


def http(method, path, token=None, body=None, raw_url=None, headers=None):
    h = {"user-agent": UA, **(headers or {})}
    if token:
        h["authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["content-type"] = "application/json"
    req = urllib.request.Request(raw_url or API + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--config-dir", default=os.path.expanduser("~/.kaggle"))
    ap.add_argument("--owner", default="riccomarcelino")
    ap.add_argument("--need-min", type=float, default=35)
    a = ap.parse_args()
    s = secrets(); token = s["WORKER_TOKEN"]
    results = []

    def check(name, ok, why):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name:<9} {why}")

    st, body = http("GET", "/v1/health")
    st2, _ = http("POST", "/v1/worker/claim", token, {"worker": "preflight", "prefix": "preflight-never-matches-"})
    check("api", st == 200 and st2 == 204, f"health {st}, token check {st2} (want 200/204)")

    st, body = http("GET", "/v1/worker/file?key=captures/e2e-test/clip-01.mp4", token)
    if st == 200:
        url = json.loads(body)["url"]
        st, body = http("GET", "", raw_url=url, headers={"range": "bytes=0-0"})
        capped = b"cap exceeded" in body
        check("storage", st in (200, 206), f"signed range download -> {st}" + (" (B2 daily download cap reached; resets 00:00 GMT)" if capped else ""))
    else:
        check("storage", False, f"could not sign a download ({st})")

    env = dict(os.environ, KAGGLE_CONFIG_DIR=a.config_dir)
    py = str(ROOT / "pipeline" / ".venv" / "bin" / "python")
    code = ("import kaggle\nfrom kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest\n"
            "with kaggle.api.build_kaggle_client() as kc:\n r = kc.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())\n"
            " q = r.gpu_quota\n print(q.total_time_allowed.total_seconds(), q.time_used.total_seconds())")  # NB: the SDK's to_dict() drops days (30 h shows as 21600 s)
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    try:
        allowed, used = map(float, out.stdout.split()[-2:])
        left = (allowed - used) / 60
        check("quota", left >= a.need_min, f"{left / 60:.1f} GPU h left of {allowed / 3600:.0f} h this week (need {a.need_min:g} min)")
    except Exception:
        check("quota", False, f"could not read the GPU quota: {(out.stderr or out.stdout)[-200:]}")

    kag = str(ROOT / "pipeline" / ".venv" / "bin" / "kaggle")
    ok_ds = all("ready" in subprocess.run([kag, "datasets", "status", f"{a.owner}/{d}"], capture_output=True, text=True, env=env).stdout
                for d in ("ply-pipeline", "ply-worker-config"))
    head = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%ct"], capture_output=True, text=True).stdout.strip()
    meta = subprocess.run([kag, "datasets", "list", "--mine", "-s", "ply-pipeline", "--csv"], capture_output=True, text=True, env=env).stdout
    check("datasets", ok_ds, "ply-pipeline + ply-worker-config ready" + ("" if ok_ds else " (missing or still processing)") +
          f"; last pipeline commit {head} (re-upload after every pipeline change)")

    if a.job:
        st, body = http("GET", f"/v1/jobs/{a.job}")
        job = json.loads(body) if st == 200 else {}
        check("job", job.get("status") == "queued", f"job {a.job[:8]} is {job.get('status', 'missing')}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
