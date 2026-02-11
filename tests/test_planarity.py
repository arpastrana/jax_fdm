from compas.datastructures import mesh_unify_cycles
import jax
import jax.numpy as jnp

from compas.datastructures import mesh_unify_cycles
from compas.geometry import Polygon
from compas.geometry import Rotation

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumMeshStructure

from jax_fdm.geometry import planarity_triangle
from jax_fdm.geometry import planarity_polygon
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
            assert jnp.allclose(planarity, 0.0), f"Planarity of {sides}-gon: {planarity:.2f}"


def test_planarity_mesh_tri():
    """
    Test the planarity of a tri mesh grid.
    """
    mesh = FDMesh.from_meshgrid(10.0, 10)
    mesh = mesh.subdivide("tri")
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
        mesh.split_edge(u, v, t=0.5, allow_boundary=True)

    mesh_unify_cycles(mesh)

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
        mesh.split_edge(u, v, t=0.5, allow_boundary=True)

    mesh_unify_cycles(mesh)

    for _ in range(3):
        key, subkey = jax.random.split(key)
        angles = jax.random.uniform(subkey, (3,), minval=0.0, maxval=2 * jnp.pi)
        _mesh = mesh.transformed(Rotation.from_euler_angles(angles))

        xyz = jnp.array([_mesh.vertex_coordinates(vkey) for vkey in _mesh.vertices()])
        structure = EquilibriumMeshStructure.from_mesh(_mesh)
        faces = structure.faces_indexed

        planarity = jnp.sum(faces_planarity(faces, xyz))
        assert jnp.allclose(planarity, 0.0), f"Planarity: {planarity}"
