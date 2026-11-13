"""The TSDF mesh (stages/mesh.py, IMPROVEMENT-PLAN 2.4, variant 3b) on the synthetic two rooms of synth.py: ray-cast depth stands in
for the splats' rendered depth, goes through dense.consistent_points exactly as export does, and is fused on a CPU by the same
torch code that runs on CUDA. The rooms are 8.1 x 4 x 2.8 m with a 10 cm partition and a 0.8 x 2 m door."""
import numpy as np
import pytest
import trimesh

from plypipe import dense
from plypipe.stages import mesh

from synth import NAV_DOOR, NAV_PARTITION, NAV_ROOMS, depth_views, look, nav_boxes, nav_walk

COLOUR = (200, 120, 40)


@pytest.fixture(scope="module")
def two_rooms():
    centers, _ = nav_walk()
    views = [(c, look(k * np.pi / 2 + 0.4 + 0.3 * i, 0.1 if k % 2 else -0.1)) for i, c in enumerate(centers[::12]) for k in range(4)]
    vs = depth_views(nav_boxes(), np.array([c for c, _ in views]), np.array([r for _, r in views]))
    for v in vs:
        v.rgb = np.broadcast_to(np.array(COLOUR, np.uint8), (v.height, v.width, 3)).copy()
    return vs


@pytest.fixture(scope="module")
def built(two_rooms, tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh")
    fusion = mesh.Fusion(voxel=0.025)
    fusion.begin(two_rooms)
    for p in dense.consistent_points(two_rooms, fuse=True):
        fusion.add(p)
    report = fusion.finish(out)
    return out, report, trimesh.load(out / "mesh.glb", force="mesh"), trimesh.load(out / "collision.glb", force="mesh")


def test_the_dense_pass_hands_the_fusion_exactly_the_depth_its_points_came_from(two_rooms):
    got = list(dense.consistent_points(two_rooms[:12], fuse=True))
    assert all(p.cam is not None and int((np.asarray(p.depth) > 0).sum()) == len(p.points) and p.rgb is not None for p in got)
    plain = list(dense.consistent_points(two_rooms[:12]))
    assert all(a.depth is None and a.cam is None and np.array_equal(a.points, b.points) for a, b in zip(plain, got))


def test_walls_are_within_two_centimetres_and_the_floor_is_at_y_zero(built):
    _, _, m, _ = built
    v = np.asarray(m.vertices)
    W, D, H = NAV_ROOMS
    for sel, axis, plane in [((v[:, 2] < 0.1) & (v[:, 0] > 0.3) & (v[:, 0] < 3.7) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 2, 0.0),
                             ((v[:, 2] > D - 0.1) & (v[:, 0] > 0.3) & (v[:, 0] < 3.7) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 2, D),
                             ((v[:, 0] < 0.1) & (v[:, 2] > 0.3) & (v[:, 2] < D - 0.3) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 0, 0.0),
                             ((v[:, 0] > W - 0.1) & (v[:, 2] > 0.3) & (v[:, 2] < D - 0.3) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5), 0, W),
                             ((v[:, 1] < 0.1) & (v[:, 0] > 0.3) & (v[:, 0] < 6.5) & ((v[:, 0] < 3.9) | (v[:, 0] > 4.2)) & (v[:, 2] > 0.3) & (v[:, 2] < D - 0.3), 1, 0.0)]:
        assert sel.sum() > 200
        assert np.abs(v[sel, axis] - plane).max() < 0.02, (axis, plane, np.abs(v[sel, axis] - plane).max())
    assert abs(float(v[:, 1].min())) < 0.02 and abs(float(v[:, 1].max()) - H) < 0.02


def test_the_door_is_an_opening_in_the_mesh_and_the_partition_beside_it_is_not(built):
    _, _, m, _ = built
    v = np.asarray(m.vertices)
    (p0, p1), (z0, z1, top) = NAV_PARTITION, NAV_DOOR[:2] + (NAV_DOOR[2],)
    on_plane = (v[:, 0] > p0 - 0.03) & (v[:, 0] < p1 + 0.03)
    gap = on_plane & (v[:, 2] > z0 + 0.08) & (v[:, 2] < z1 - 0.08) & (v[:, 1] > 0.1) & (v[:, 1] < top - 0.1)
    solid = on_plane & (v[:, 2] > 0.1) & (v[:, 2] < z0 - 0.2) & (v[:, 1] > 0.3) & (v[:, 1] < 2.5)
    assert gap.sum() == 0 and solid.sum() > 100


def test_the_collision_mesh_is_within_its_budget_and_within_five_centimetres_of_the_mesh(built):
    out, report, m, c = built
    assert report["built"] and len(m.faces) <= mesh.MESH_FACES * 1.05 and len(c.faces) <= mesh.COLLISION_FACES * 1.05, report
    assert len(c.faces) < len(m.faces) or len(m.faces) <= mesh.COLLISION_FACES * 1.05
    from scipy.spatial import cKDTree
    a, b = m.sample(600_000), c.sample(600_000)
    tree = cKDTree(a)
    # a collision triangle may bridge the hole of a patch no camera saw: 6 cm at most, on 0.05 % of the surface
    near = [tree.query(x)[0] for x in (c.vertices, b)]
    assert all(np.percentile(d, 99.9) < 0.05 and d.max() < 0.07 for d in near), [d.max() for d in near]
    assert (out / "mesh.glb").stat().st_size > 0 and (out / "collision.glb").stat().st_size > 0


def test_vertex_colours_are_the_fused_colours_and_normals_face_the_room(built):
    _, _, m, _ = built
    col = np.asarray(m.visual.vertex_colors)[:, :3].astype(int)
    assert np.abs(col - np.array(COLOUR)).max() <= 1
    near = m.triangles_center[:, 2] < 0.05            # the wall z = 0: its normals point into the room, +z
    assert near.sum() > 20 and float(m.face_normals[near, 2].mean()) > 0.9


def test_a_plane_seen_by_one_camera_is_meshed_at_its_distance_and_decimation_keeps_it_flat():
    from plypipe.stages.mesh import Tsdf, decimate
    K = np.array([[120.0, 0, 80], [0, 120.0, 60], [0, 0, 1]])
    depth = np.full((120, 160), 3.0, np.float32)
    ts = Tsdf(0.03)
    ts.integrate(depth, K, np.eye(3), np.zeros(3))
    v, c, f = ts.extract()
    assert len(f) > 1000 and np.abs(v[:, 2] - 3.0).max() < 0.01 and (np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2] < 0).mean() > 0.99
    dv, df, _ = decimate(v, f, None, 40)
    assert len(df) <= 42 and np.abs(dv[:, 2] - 3.0).max() < 0.01


def test_the_voxel_is_chosen_by_the_size_of_the_scene_and_a_chapel_fits_in_memory():
    assert mesh.choose_voxel((8, 3, 4)) == 0.02
    chapel = mesh.choose_voxel((25, 12, 8))
    assert 0.02 <= chapel <= 0.03
    assert mesh.choose_voxel((200, 40, 30)) == 0.04
