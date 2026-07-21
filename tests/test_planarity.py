import jax
import jax.numpy as jnp

from compas.geometry import Polygon
from compas.geometry import Rotation
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.geometry import planarity_polygon
from jax_fdm.geometry import planarity_triangle
from jax_fdm.goals.mesh.planarity import face_planarity
from jax_fdm.goals.mesh.planarity import faces_planarity


def test_planarity_polygons():
    """
    Test the planarity of different polygons.
    """
    key = jax.random.PRNGKey(1701)
    for sides in range(3, 10):
        polygon = Polygon.from_sides_and_radius_xy(sides, 1.0)
        for _ in range(3):
            key, subkey = jax.random.split(key)
            angles = jax.random.uniform(subkey, (3,), minval=0.0, maxval=2 * jnp.pi)
            polygon = polygon.transformed(Rotation.from_euler_angles(angles))
            xyz = jnp.array(polygon.points)
            if sides == 3:
                planarity = planarity_triangle(xyz)
            else:
                planarity = planarity_polygon(xyz)
            assert jnp.allclose(planarity, 0.0), (
                f"Planarity of {sides}-gon: {planarity:.2f}"
            )


def test_planarity_mesh_tri():
    """
    Test the planarity of a tri mesh grid.
    """
    mesh = FDMesh.from_meshgrid(10.0, 10)
    mesh = mesh.subdivided(scheme="tri")
    key = jax.random.PRNGKey(1701)

    for _ in range(3):
        key, subkey = jax.random.split(key)
        angles = jax.random.uniform(subkey, (3,), minval=0.0, maxval=2 * jnp.pi)
        print(angles)

        mesh = mesh.transformed(Rotation.from_euler_angles(angles))

        xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        structure = EquilibriumMeshStructure.from_mesh(mesh)
        faces = structure.faces_indexed

        planarity = jnp.sum(faces_planarity(faces, xyz))
        assert jnp.allclose(planarity, 0.0), f"Planarity: {planarity}"


def test_planarity_mesh_quad():
    """
    Test the planarity of a quad mesh grid.
    """
    mesh = FDMesh.from_meshgrid(10.0, 10)
    key = jax.random.PRNGKey(1701)

    for _ in range(3):
        key, subkey = jax.random.split(key)
        angles = jax.random.uniform(subkey, (3,), minval=0.0, maxval=2 * jnp.pi)
        mesh = mesh.transformed(Rotation.from_euler_angles(angles))

        xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        structure = EquilibriumMeshStructure.from_mesh(mesh)
        faces = structure.faces_indexed

        planarity = jnp.sum(faces_planarity(faces, xyz))
        assert jnp.allclose(planarity, 0.0), f"Planarity: {planarity}"


def test_planarity_mesh_ngon_flat():
    """
    Test the planarity of a mesh grid with faces with 4, 5, and 6 vertices.
    """
    mesh = FDMesh.from_meshgrid(10.0, 5)
    key = jax.random.PRNGKey(1701)

    for u, v in mesh.edges_on_boundary():
        mesh.split_edge((u, v), t=0.5, allow_boundary=True)

    mesh.unify_cycles()

    for _ in range(3):
        key, subkey = jax.random.split(key)
        angles = jax.random.uniform(subkey, (3,), minval=0.0, maxval=2 * jnp.pi)
        _mesh = mesh.transformed(Rotation.from_euler_angles(angles))

        xyz = jnp.array([_mesh.vertex_coordinates(vkey) for vkey in _mesh.vertices()])
        structure = EquilibriumMeshStructure.from_mesh(_mesh)
        faces = structure.faces_indexed

        planarity = jnp.sum(faces_planarity(faces, xyz))
        assert jnp.allclose(planarity, 0.0), f"Planarity: {planarity}"


def test_planarity_mesh_quad_barrel():
    """
    Test the planarity of a mesh grid with faces with 4, 5, and 6 vertices.
    """
    mesh = FDMesh.from_obj("tests/data/barrel.obj")

    key = jax.random.PRNGKey(1701)

    for u, v in mesh.edges_on_boundary():
        mesh.split_edge((u, v), t=0.5, allow_boundary=True)

    mesh.unify_cycles()

    for _ in range(3):
        key, subkey = jax.random.split(key)
        angles = jax.random.uniform(subkey, (3,), minval=0.0, maxval=2 * jnp.pi)
        _mesh = mesh.transformed(Rotation.from_euler_angles(angles))

        xyz = jnp.array([_mesh.vertex_coordinates(vkey) for vkey in _mesh.vertices()])
        structure = EquilibriumMeshStructure.from_mesh(_mesh)
        faces = structure.faces_indexed

        planarity = jnp.sum(faces_planarity(faces, xyz))
        assert jnp.allclose(planarity, 0.0), f"Planarity: {planarity}"


def test_planarity_far_from_origin():
    """
    Test that planarity does not drift when the mesh sits far from the origin.
    """
    xyz = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
    )
    face = jnp.array([0, 1, 2, 3])
    planarity = face_planarity(face, xyz)

    for shift in (1e6, 1e8, 1e12):
        planarity_shifted = face_planarity(face, xyz + shift)
        assert jnp.allclose(planarity, planarity_shifted), f"Drift at {shift:.0e}"


def test_planarity_face_padding_invariant():
    """
    Test that -1 padding does not change a face's planarity or its gradient.
    """
    xyz = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0],
        ],
    )
    face = jnp.array([0, 1, 2, 3])
    face_padded = jnp.array([0, 1, 2, 3, -1, -1])

    planarity = face_planarity(face, xyz)
    planarity_padded = face_planarity(face_padded, xyz)
    assert planarity > 0.0
    assert jnp.allclose(planarity, planarity_padded)

    grad = jax.grad(lambda x: face_planarity(face, x))(xyz)
    grad_padded = jax.grad(lambda x: face_planarity(face_padded, x))(xyz)
    assert not jnp.any(jnp.isnan(grad_padded))
    assert jnp.allclose(grad, grad_padded)
    assert jnp.allclose(grad[-1], 0.0)


def test_planarity_per_edge_mean():
    """
    Test that face planarity averages over edges, not sums over them.
    """

    def corrugated_ngon(num_sides, height):
        angles = jnp.arange(num_sides) * 2.0 * jnp.pi / num_sides
        z = jnp.where(jnp.arange(num_sides) % 2 == 0, height, -height)
        return jnp.stack([jnp.cos(angles), jnp.sin(angles), z], axis=-1)

    # the mean absolute cosine equals the raw cosine sum divided by edge count
    for num_sides in (4, 6, 8):
        xyz = corrugated_ngon(num_sides, 0.2)
        face = jnp.arange(num_sides)
        planarity = face_planarity(face, xyz)
        planarity_sum = planarity_polygon(xyz)
        assert jnp.allclose(planarity, planarity_sum / num_sides)

    # comparable per-edge deviations stay same-order across face degrees
    quad = face_planarity(jnp.arange(4), corrugated_ngon(4, 0.2))
    hexa = face_planarity(jnp.arange(6), corrugated_ngon(6, 0.2))
    octa = face_planarity(jnp.arange(8), corrugated_ngon(8, 0.2))
    assert hexa < 2.0 * quad
    assert octa < 3.0 * quad
