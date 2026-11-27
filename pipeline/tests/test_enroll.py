"""plypipe.enroll: pair, wait for approval, keep the token private; and the worker picking it up."""

import json
import os
import stat

import pytest

from plypipe import enroll


def test_pair_waits_for_approval_and_saves_the_token_private(monkeypatch, tmp_path):
    monkeypatch.setenv("PLY_MACHINE_FILE", str(tmp_path / "worker.json"))
    monkeypatch.setattr(enroll, "specs", lambda: {"why": "GTX 1650 test", "trainable": True})
    answers = iter([(201, {"name": "pc-lab-ab12", "code": "ABCD-EFGH", "poll": "p" * 64, "expiresInSec": 600}),
                    (200, {"status": "pending"}), (200, {"status": "pending"}),
                    (200, {"status": "active", "name": "pc-lab-ab12", "token": "pm_pc-lab-ab12." + "a" * 64})])
    calls = []
    monkeypatch.setattr(enroll, "post", lambda api, path, body, timeout=60: (calls.append((path, body)), next(answers))[1])
    res = enroll.pair("https://api.test", every=0, sleep=lambda s: None, out=lambda s: None)
    assert res["name"] == "pc-lab-ab12"
    assert calls[0][0] == "/v1/machines/pair" and calls[0][1]["specs"]["why"] == "GTX 1650 test"
    assert all(c[1] == {"name": "pc-lab-ab12", "poll": "p" * 64} for c in calls[1:])
    saved = json.loads((tmp_path / "worker.json").read_text())
    assert saved["token"].startswith("pm_pc-lab-ab12.") and saved["api"] == "https://api.test"
    assert stat.S_IMODE(os.stat(tmp_path / "worker.json").st_mode) == 0o600
    assert enroll.load()["name"] == "pc-lab-ab12"
    assert enroll.main(["--check"]) == 0


def test_a_refused_pairing_stops(monkeypatch, tmp_path):
    monkeypatch.setenv("PLY_MACHINE_FILE", str(tmp_path / "worker.json"))
    monkeypatch.setattr(enroll, "specs", lambda: {"why": ""})
    monkeypatch.setattr(enroll, "post", lambda *a, **k: (429, {"error": "too many pairing requests"}))
    with pytest.raises(SystemExit, match="too many"):
        enroll.pair("https://api.test", every=0, sleep=lambda s: None, out=lambda s: None)
    assert enroll.main(["--check"]) == 1


def test_worker_uses_the_paired_token_and_stops_when_revoked(monkeypatch, tmp_path):
    from plypipe import worker
    monkeypatch.setenv("PLY_MACHINE_FILE", str(tmp_path / "worker.json"))
    monkeypatch.delenv("PLY_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("PLY_API", raising=False)
    enroll.save({"api": "https://api.test", "name": "pc-lab-ab12", "token": "pm_pc-lab-ab12." + "a" * 64})
    monkeypatch.setenv("PLY_DEVICE_VRAM_GB", "4")
    monkeypatch.setenv("PLY_LIVE_SNAPSHOTS", "")  # worker.main setdefaults it; scoped here so it cannot leak into later tests
    seen = {}

    def req(self, method, path, body=None, ctype=None, tries=5):
        seen.update(token=self.token, base=self.base, body=body)
        raise RuntimeError(f"{method} {path}: 401 b'unauthorized'")

    monkeypatch.setattr(worker.Api, "req", req)
    assert worker.main(["--work", str(tmp_path / "w")]) == 3
    assert seen["token"].startswith("pm_pc-lab-ab12.") and seen["base"] == "https://api.test"
    assert seen["body"]["worker"] == "pc-lab-ab12" and seen["body"]["specs"]["packed"] is True
