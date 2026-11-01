#!/usr/bin/env python3
# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Mint or check a Ply invite code.

Ply has no sign-up. An invite is `label.expiry.signature`, where the signature is an HMAC of
`label.expiry` with the app key the Worker already holds, so the Worker can verify one with Web Crypto
and no database. Expiry bounds a leaked code; rotating APP_KEY revokes every outstanding invite.

    python3 scripts/invite.py new --label toeesh --days 14
    python3 scripts/invite.py check toeesh.1767225600.3f9a...
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import re
import sys
import time
from pathlib import Path

SECRETS = Path(__file__).resolve().parents[1] / "cloud" / ".secrets"
CODE = re.compile(r"^([a-z0-9][a-z0-9-]{1,23})\.(\d{10})\.([0-9a-f]{32})$")


def app_key() -> str:
    for line in SECRETS.read_text().splitlines():
        if line.startswith("APP_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no APP_KEY in {SECRETS}")


def sign(key: str, label: str, expiry: int) -> str:
    return hmac.new(key.encode(), f"{label}.{expiry}".encode(), hashlib.sha256).hexdigest()[:32]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new", help="mint a code")
    new.add_argument("--label", required=True, help="who it is for, lowercase, no spaces")
    new.add_argument("--days", type=int, default=14, help="days until it stops working (default 14)")
    chk = sub.add_parser("check", help="verify a code against the local app key")
    chk.add_argument("code")
    a = ap.parse_args(argv)

    key = app_key()
    if a.cmd == "new":
        label = a.label.lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,23}", label):
            raise SystemExit("label must be lowercase letters, digits or dashes, 2 to 24 characters")
        exp = int(time.time()) + a.days * 86400
        code = f"{label}.{exp}.{sign(key, label, exp)}"
        print(code)
        print(f"valid until {time.strftime('%Y-%m-%d %H:%M', time.localtime(exp))}", file=sys.stderr)
        print("send the code, not this terminal: it is a bearer key until it expires", file=sys.stderr)
        return 0

    m = CODE.match(a.code.strip())
    if not m:
        print("not an invite code (want label.expiry.signature)", file=sys.stderr)
        return 1
    label, exp, mac = m.groups()
    if int(exp) * 1000 < time.time() * 1000:
        print(f"expired {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(exp)))}", file=sys.stderr)
        return 1
    ok = hmac.compare_digest(sign(key, label, int(exp)), mac)
    print("valid" if ok else "signature does not match this app key", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
