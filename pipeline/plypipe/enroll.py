# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Pair this PC with the Ply API as a worker (cloud/src/machines.ts). Standard library only.

    python -m plypipe.enroll            # shows a code; the team approves it; the token lands in ~/.ply/worker.json
    python -m plypipe.enroll --check    # is this PC paired? (exit 0 yes, 1 no)

No secret ships with the code: the PC asks for a pairing code, shows it, and polls until someone with the team key
approves it (`python scripts/plyctl.py machines approve CODE`). The token it then receives is for this PC alone, can
only take jobs assigned to it by name, and stops working when the team revokes it.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API = "https://ply-api.ricco-ply.workers.dev"
UA = {"user-agent": "ply-worker/1.0 (+https://ricco.codes/ply)", "content-type": "application/json"}


def machine_file() -> Path:
    return Path(os.environ.get("PLY_MACHINE_FILE") or Path.home() / ".ply" / "worker.json")


def load() -> dict | None:
    try:
        d = json.loads(machine_file().read_text())
        return d if d.get("token") and d.get("name") else None
    except (OSError, ValueError):
        return None


def save(d: dict) -> Path:
    path = machine_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return path


def post(api: str, path: str, body: dict, timeout: float = 60) -> tuple[int, dict]:
    req = urllib.request.Request(api.rstrip("/") + path, data=json.dumps(body).encode(), method="POST", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def specs() -> dict:
    from plypipe import device
    return device.device_plan(device.probe(Path.home()))


def banner(code: str, name: str, why: str) -> None:
    line = "=" * 58
    print(f"\n{line}\n  PAIRING CODE:   {code}\n{line}")
    print(f"  This PC will join Ply as '{name}'.")
    print("  Send the code to the Ply team. They approve it with:")
    print(f"      python scripts/plyctl.py machines approve {code}")
    print(f"  This machine: {why}")
    print("  Waiting for approval (the code is valid for 24 hours; Ctrl+C to stop)...\n", flush=True)


def pair(api: str, every: float = 5.0, sleep=time.sleep, out=print) -> dict:
    dp = specs()
    status, got = post(api, "/v1/machines/pair", {"hint": socket.gethostname(), "specs": dp})
    if status != 201:
        raise SystemExit(f"[pair] the API refused to pair ({status}): {got.get('error', got)}")
    banner(got["code"], got["name"], dp.get("why", ""))
    deadline = time.time() + float(got.get("expiresInSec", 86400))
    last = 0.0
    while time.time() < deadline:
        sleep(every)
        try:
            status, res = post(api, "/v1/machines/poll", {"name": got["name"], "poll": got["poll"]})
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            out(f"[pair] network hiccup ({e}); retrying")
            continue
        if status == 200 and res.get("status") == "active":
            path = save({"api": api, "name": res["name"], "token": res["token"], "pairedAt": int(time.time())})
            out(f"[pair] approved: this PC is worker '{res['name']}' (token saved to {path}, readable only by you)")
            return res
        if status != 200 or res.get("status") not in ("pending", "approved"):
            raise SystemExit(f"[pair] pairing ended: {res.get('status') or res.get('error') or status}")
        if time.time() - last > 300:
            out(f"[pair] still waiting for approval of {got['code']}")
            last = time.time()
    raise SystemExit("[pair] the code expired: run pairing again")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m plypipe.enroll")
    ap.add_argument("--api", default=os.environ.get("PLY_API") or DEFAULT_API)
    ap.add_argument("--check", action="store_true", help="exit 0 when this PC already has a worker token")
    ap.add_argument("--again", action="store_true", help="pair again even if a token is saved (e.g. after a revoke)")
    a = ap.parse_args(argv)
    have = load()
    if a.check:
        print(f"paired as {have['name']}" if have else "not paired")
        return 0 if have else 1
    if have and not a.again:
        print(f"[pair] already paired as '{have['name']}' ({machine_file()}); --again to pair anew")
        return 0
    pair(a.api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
