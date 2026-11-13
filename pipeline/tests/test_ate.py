# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Absolute trajectory error: zero for the same path under any similarity, the known size of a known error otherwise."""

import numpy as np
import pytest

from plypipe import ate, geometry


def _path(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {f"{i:04d}.jpg": p for i, p in enumerate(np.cumsum(rng.normal(0, 0.1, (n, 3)), axis=0))}


def _similarity(points, scale, yaw, shift):
    R = geometry.rot_y(yaw) @ geometry.rot_y(0.0)
    tilt = np.array([[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]])
    return {k: scale * (tilt @ R @ v) + shift for k, v in points.items()}


@pytest.mark.parametrize("scale, yaw, shift", [(1.0, 0.0, [0, 0, 0]), (3.7, 2.1, [10, -4, 25]), (0.02, -1.3, [-1e3, 5, 0.5])])
def test_the_error_of_an_identical_path_is_zero_under_any_similarity(scale, yaw, shift):
    ref = _path()
    got = ate.ate(_similarity(ref, scale, yaw, np.array(shift, dtype=float)), ref)
    assert got["rmse"] < 1e-9 and got["median"] < 1e-9 and got["max"] < 1e-9 and got["matched"] == 200
    assert got["scale"] == pytest.approx(1 / scale)


def test_the_error_matches_a_known_noise_level_in_the_reference_units():
    ref = _path(4000, 1)
    sigma = 0.05  # per axis; the fit spends 7 of the 12,000 degrees of freedom
    noise = np.random.default_rng(2).normal(0, sigma, (4000, 3))
    est = {k: v + n for (k, v), n in zip(ref.items(), noise)}
    got = ate.ate(est, ref)
    assert got["rmse"] == pytest.approx(sigma * np.sqrt(3), rel=0.02)
    assert got["median"] == pytest.approx(sigma * 1.538, rel=0.04)  # median of a chi distribution with 3 degrees of freedom


def test_one_misplaced_view_shows_in_the_rmse_and_max_but_not_the_median():
    ref = _path()
    est = dict(ref)
    est["0100.jpg"] = est["0100.jpg"] + np.array([1.0, 0.0, 0.0])
    got = ate.ate(est, ref)
    assert got["median"] < 0.01 and 0.9 < got["max"] <= 1.0 and got["rmse"] == pytest.approx(1 / np.sqrt(200), rel=0.15)


def test_views_are_matched_by_name_and_the_unmatched_are_counted_not_scored():
    ref = _path()
    est = {k: v for k, v in ref.items() if int(k[:4]) % 2 == 0}
    est["extra.jpg"] = np.zeros(3)
    got = ate.ate(est, ref)
    assert got["matched"] == 100 and got["estPosed"] == 101 and got["refPosed"] == 200 and got["rmse"] < 1e-9


def test_too_few_common_views_cannot_be_aligned():
    with pytest.raises(ValueError, match="only 2 views"):
        ate.ate({"a": np.zeros(3), "b": np.ones(3)}, {"a": np.zeros(3), "b": np.ones(3), "c": np.ones(3)})
