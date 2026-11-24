"""Filling the floor the camera never saw: enclosed holes get floor copied from beside them; gaps at the edge do not."""
import numpy as np

from plypipe import clean, fill


def write_floor(path, hole=True, rng=np.random.default_rng(0)):
    """A 4 x 4 m floor of small flat splats (normal +y), a 40 cm hole in the middle, and a notch at one edge."""
    x, z = rng.uniform(-2, 2, 60000), rng.uniform(-2, 2, 60000)
    keep = ~((np.abs(x) < 0.2) & (np.abs(z) < 0.2)) if hole else np.ones(len(x), bool)
    keep &= ~((x > 1.6) & (np.abs(z) < 0.3))  # a notch open to the edge: outside, not a hole
    x, z = x[keep], z[keep]; n = len(x)
    s = np.log(np.c_[np.full(n, .02), np.full(n, .002), np.full(n, .02)])   # thin along y: flat on the floor
    cols = {"x": x, "y": np.zeros(n), "z": z, "f_dc_0": rng.uniform(-1, 1, n), "f_dc_1": np.zeros(n), "f_dc_2": np.zeros(n),
            "opacity": np.full(n, 3.0), "scale_0": s[:, 0], "scale_1": s[:, 1], "scale_2": s[:, 2],
            "rot_0": np.ones(n), "rot_1": np.zeros(n), "rot_2": np.zeros(n), "rot_3": np.zeros(n)}
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {p}\n" for p in cols) + "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode()); np.stack([np.asarray(v, np.float32) for v in cols.values()], 1).tofile(f)
    return n


def test_an_enclosed_hole_is_filled_and_an_open_edge_is_not(tmp_path):
    n = write_floor(tmp_path / "in.ply")
    rep = fill.fill_floors(tmp_path / "in.ply", tmp_path / "out.ply")
    assert rep["floors"][0]["holes"] == 1 and 0.1 < rep["floors"][0]["filledM2"] < 0.25, rep
    _, cols = clean.read_ply(tmp_path / "out.ply")
    assert len(cols["x"]) == n + rep["added"]
    x, z = cols["x"], cols["z"]
    assert ((np.abs(x) < 0.15) & (np.abs(z) < 0.15)).sum() > 100           # the hole has floor in it now
    assert ((x > 1.7) & (np.abs(z) < 0.2)).sum() == 0                       # the notch at the edge stays open
    assert np.allclose(np.unique(cols["scale_1"][n:]), np.log(np.float32(.002)), atol=1e-4)  # copies keep their shape


def test_a_floor_without_holes_is_left_alone(tmp_path):
    write_floor(tmp_path / "in.ply", hole=False)
    rep = fill.fill_floors(tmp_path / "in.ply", tmp_path / "out.ply")
    assert rep["added"] == 0 and not (tmp_path / "out.ply").exists()
