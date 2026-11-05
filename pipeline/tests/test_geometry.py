import numpy as np

from plypipe import geometry


def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(geometry.mat_to_quat(geometry.quat_to_mat(q)), q, atol=1e-9)


def test_rotation_between_including_opposite():
    for a, b in [([0, 0, 1], [0, 1, 0]), ([1, 2, 3], [-1, 0, 2]), ([0, 1, 0], [0, -1, 0])]:
        R = geometry.rotation_between(np.array(a, float), np.array(b, float))
        assert np.allclose(R @ (np.array(a) / np.linalg.norm(a)), np.array(b) / np.linalg.norm(b), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_transform_pose_keeps_projections():
    rng = np.random.default_rng(3)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    t = rng.normal(size=3)
    X = rng.normal(size=(20, 3)) * 3 + [0, 0, 6]
    T = np.eye(4); T[:3, :3] = 2.5 * geometry.quat_to_mat(rng.normal(size=4)); T[:3, 3] = [1, -2, 3]
    q2, t2 = geometry.transform_pose(q, t, T)
    x1 = X @ geometry.quat_to_mat(q).T + t
    x2 = geometry.apply(T, X) @ geometry.quat_to_mat(q2).T + t2
    assert np.allclose(x1[:, :2] / x1[:, 2:], x2[:, :2] / x2[:, 2:], atol=1e-9)


def test_ransac_picks_floor_not_table_or_ceiling():
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(0, 6, 300), np.zeros(300), rng.uniform(0, 4, 300)]
    # a heavily textured table top with MORE points than the floor
    table = np.c_[rng.uniform(2, 3, 900), np.full(900, 0.75), rng.uniform(1, 2, 900)]
    ceiling = np.c_[rng.uniform(0, 6, 2000), np.full(2000, 2.8), rng.uniform(0, 4, 2000)]
    pts = np.concatenate([floor, table, ceiling]) + rng.normal(0, 0.005, (3200, 3))
    cams = np.c_[rng.uniform(1, 5, 30), np.full(30, 1.45), rng.uniform(1, 3, 30)]
    plane = geometry.ransac_floor(pts, cams, np.array([0.05, 1, 0.02]))
    assert plane.normal[1] > 0.999
    assert abs(plane.offset) < 0.02
    assert plane.inliers[:300].mean() > 0.9


def test_dominant_yaw_recovers_room_axes():
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 1, 2000)
    walls = np.concatenate([np.c_[6 * t[:500], np.zeros(500)], np.c_[np.full(500, 6), 4 * t[500:1000]],
                            np.c_[6 * t[1000:1500], np.full(500, 4)], np.c_[np.zeros(500), 4 * t[1500:]]])
    for deg in (0, 17, 38, -25, 80):
        a = np.radians(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        yaw = geometry.dominant_yaw(walls @ R.T)
        # Manhattan ambiguity: equal modulo 90 deg
        diff = (np.degrees(yaw) - deg + 45) % 90 - 45
        assert abs(diff) < 3, (deg, np.degrees(yaw))


def _cams(yaws_deg, pitches_deg, roll_deg=0.0):
    """Camera up and forward vectors in a +y-up world, for the given yaw / pitch (deg) and a fixed roll."""
    ups, fwds = [], []
    for yaw, pitch in zip(np.radians(yaws_deg), np.radians(pitches_deg)):
        f = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
        r = np.cross(f, [0.0, 1.0, 0.0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        ups.append(c * u + s * r); fwds.append(f)
    return np.array(ups), np.array(fwds)


def test_gravity_from_level_cameras_ignores_pitch():
    """Facing one end and looking up at the vault tilts the mean up vector; the cameras' right axes stay level."""
    rng = np.random.default_rng(0)
    yaws = rng.uniform(-70, 70, 400)
    pitches = np.where(rng.random(400) < 0.4, rng.uniform(30, 60, 400), rng.uniform(-15, 10, 400))
    ups, fwds = _cams(yaws, pitches, roll_deg=2.0)
    mean = ups.mean(0) / np.linalg.norm(ups.mean(0))
    g, how = geometry.gravity_guess(ups, fwds)
    deg = lambda a: np.degrees(np.arccos(np.clip(a @ [0, 1.0, 0], -1, 1)))
    assert how == "camera-level" and deg(g) < 2.5 and deg(mean) > 8


def test_gravity_falls_back_to_mean_up_when_the_camera_never_turned():
    ups, fwds = _cams(np.zeros(50), np.linspace(-20, 40, 50))
    g, how = geometry.gravity_guess(ups, fwds)
    assert how == "mean-up" and np.allclose(g, ups.mean(0) / np.linalg.norm(ups.mean(0)))
