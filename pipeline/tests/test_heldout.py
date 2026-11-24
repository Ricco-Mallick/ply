"""Held-out photos for the viewer's photo-vs-twin check: which views, and the camera conversion to three.js."""
import numpy as np

from plypipe import heldout


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_qvec_wxyz(R):
    x, y, z, w = heldout.R_to_quat_xyzw(R)
    return np.array([w, x, y, z])


def test_a_colmap_camera_lands_where_three_expects_it():
    # an OpenCV camera at (1, 1.5, 2) looking along world +x with world +y up (image y down = world -y)
    R_c2w_cv = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)   # columns: cam x, y, z in world
    C = np.array([1.0, 1.5, 2.0])
    R_cw = R_c2w_cv.T
    pos, q = heldout.three_pose(R_to_qvec_wxyz(R_cw), -R_cw @ C)
    assert np.allclose(pos, C, atol=1e-4)
    R = quat_to_R(q)
    assert np.allclose(R @ [0, 0, -1], [1, 0, 0], atol=1e-5)   # three looks down -z: here world +x
    assert np.allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-5)    # and keeps +y up


def test_pick_follows_gsplat_test_every_and_spreads():
    names = [f"{i:04d}.jpg" for i in range(300)]
    assert heldout.pick(names, 0) == [] and heldout.pick(names, 1) == []
    got = heldout.pick(list(reversed(names)), 30)              # order of the input does not matter
    val = [f"{i:04d}.jpg" for i in range(0, 300, 30)]
    assert len(got) == heldout.MAX_VIEWS and set(got) <= set(val) and got[0] == val[0] and got[-1] == val[-1]
