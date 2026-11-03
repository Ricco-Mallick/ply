# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import gzip
import subprocess

import numpy as np
import pytest

from plypipe import spz
from plypipe.stages import export
from test_export import PROPS, pytestmark, write_gs_ply  # noqa: F401  (same npx skip)


def _sh_bytes(path, n, k):
    raw = gzip.decompress(path.read_bytes())
    return np.frombuffer(raw, np.uint8, n * k * 3, 16 + n * 20).reshape(n, k, 3).astype(int)


def _ply(tmp_path, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ply = tmp_path / f"in{seed}.ply"
    write_gs_ply(ply, rng.uniform(-3, 3, (n, 3)), rng)
    raw = ply.read_bytes()
    end = raw.index(b"end_header\n") + 11
    return ply, np.frombuffer(raw[end:], np.float32).reshape(n, len(PROPS))


def _to_spz(ply, out, *extra):
    subprocess.run(["npx", "-y", export.SPLAT_TRANSFORM, "-w", "--spz-version", "3", str(ply), *extra, str(out)],
                   check=True, capture_output=True)


@pytest.mark.parametrize("bits", [8, 6])
def test_restore_sh_writes_the_coefficients_at_the_asked_precision(tmp_path, bits):
    ply, data = _ply(tmp_path)
    out = tmp_path / "s.spz"
    _to_spz(ply, out)
    rounded = _sh_bytes(out, len(data), 15)
    rep = spz.restore_sh(out, ply, bits=bits)
    assert rep["restored"] and rep["shDegree"] == 3
    got = _sh_bytes(out, len(data), 15)
    want = data[:, 9:54].reshape(-1, 3, 15).transpose(0, 2, 1)  # PLY channel-major -> SPZ coefficient-major
    step = 2 ** (8 - bits)
    assert np.abs((got - 128) / 128 - want).max() <= step / 256 + 1e-6
    assert np.abs((got - 128) / 128 - want).mean() < np.abs((rounded - 128) / 128 - want).mean()
    # everything before the SH block is what splat-transform wrote
    before = gzip.decompress(out.read_bytes())[:16 + len(data) * 20]
    _to_spz(ply, tmp_path / "again.spz")
    assert before == gzip.decompress((tmp_path / "again.spz").read_bytes())[:16 + len(data) * 20]
    # and the file still reads back
    subprocess.run(["npx", "-y", export.SPLAT_TRANSFORM, "-w", str(out), str(tmp_path / "back.ply")], check=True,
                   capture_output=True)


def test_restore_sh_keeps_lower_bands_of_a_filtered_file(tmp_path):
    ply, data = _ply(tmp_path)
    out = tmp_path / "h1.spz"
    _to_spz(ply, out, "-H", "1")
    assert spz.restore_sh(out, ply)["restored"]
    got = _sh_bytes(out, len(data), 3)
    want = data[:, 9:54].reshape(-1, 3, 15)[:, :, :3].transpose(0, 2, 1)
    assert np.abs((got - 128) / 128 - want).max() <= 1 / 256 + 1e-6


def test_restore_sh_leaves_a_file_from_other_splats_alone(tmp_path):
    ply, _ = _ply(tmp_path)
    other, _ = _ply(tmp_path, seed=1)
    out = tmp_path / "s.spz"
    _to_spz(ply, out)
    keep = out.read_bytes()
    rep = spz.restore_sh(out, other)
    assert not rep["restored"] and "different splats" in rep["why"]
    assert out.read_bytes() == keep
    _to_spz(ply, tmp_path / "h0.spz", "-H", "0")
    assert not spz.restore_sh(tmp_path / "h0.spz", ply)["restored"]
