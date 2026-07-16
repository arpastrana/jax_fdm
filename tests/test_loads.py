"""
Characterization tests for the mesh tributary load-assembly model.
"""

import jax.numpy as jnp
import pytest

from jax_fdm.equilibrium import EquilibriumMeshStructureSparse
from jax_fdm.equilibrium import calculate_faces_load
from jax_fdm.equilibrium import nodes_load_from_edges
from jax_fdm.equilibrium import nodes_load_from_faces
from jax_fdm.geometry import area_triangle
from jax_fdm.geometry import length_vector
from jax_fdm.geometry import normal_polygon
from jax_fdm.geometry import normalize_vector
from jax_fdm.geometry import polygon_lcs

PZ = 2.0
# A general load vector with all three components nonzero, so the local-load
# tests exercise the in-plane (x, y) terms and not only the normal (z) term.
LOAD = [0.5, -0.3, 2.0]
MESHES = [
    "meshgrid_mesh",
    "meshgrid_mesh_fine",
    "tetra_mesh",
    "octa_mesh",
    "irregular_mesh",
]


def _structure_xyz(mesh):
    """
    Build a sparse equilibrium structure and its vertex coordinate array.
    """
    structure = EquilibriumMeshStructureSparse.from_mesh(mesh)
    xyz = jnp.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])

    return structure, xyz


def _total_face_area(structure, xyz):
    """
    Sum the centroid-fan area of every face, matching how the load model defines
    tributary area, so the invariant holds for non-planar faces too.
    """
    total = 0.0
    for face in structure.faces_indexed:
        indices = jnp.array([int(i) for i in face if int(i) >= 0])
        face_xyz = xyz[indices, :]
        centroid = jnp.mean(face_xyz, axis=0)
        count = face_xyz.shape[0]
        for i in range(count):
            triangle = jnp.vstack([face_xyz[i], face_xyz[(i + 1) % count], centroid])
            total += jnp.squeeze(area_triangle(triangle))

    return total


def _total_edge_length(structure, xyz):
    """
    Sum the length of every edge from the structure connectivity.
    """
    return jnp.sum(length_vector(structure.connectivity @ xyz))


# ==============================================================================
# Load conservation invariants (golden-free, COMPAS-free)
# ==============================================================================


@pytest.mark.parametrize("mesh_name", MESHES)
def test_faces_load_resultant_conserved(mesh_name, request):
    """
    The assembled node loads sum to the total applied face load (pz times area).
    """
    mesh = request.getfixturevalue(mesh_name)
    mesh.faces_loads([0.0, 0.0, PZ])
    structure, xyz = _structure_xyz(mesh)

    faces_load = jnp.asarray(mesh.faces_loads())
    nodes_load = nodes_load_from_faces(xyz, faces_load, structure)

    resultant = jnp.sum(nodes_load, axis=0)
    expected = PZ * _total_face_area(structure, xyz)

    assert jnp.allclose(resultant[2], expected)
    assert jnp.allclose(resultant[:2], 0.0)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_edges_load_resultant_conserved(mesh_name, request):
    """
    The assembled node loads sum to the total applied edge load (pz times length).
    """
    mesh = request.getfixturevalue(mesh_name)
    mesh.edges_loads([0.0, 0.0, PZ])
    structure, xyz = _structure_xyz(mesh)

    edges_load = jnp.asarray(mesh.edges_loads())
    nodes_load = nodes_load_from_edges(xyz, edges_load, structure)

    resultant = jnp.sum(nodes_load, axis=0)
    expected = PZ * _total_edge_length(structure, xyz)

    assert jnp.allclose(resultant[2], expected)
    assert jnp.allclose(resultant[:2], 0.0)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_local_faces_load_preserves_input_components(mesh_name, request):
    """
    A local (follower) face load carries the full x, y, z input into each face
    frame, recoverable by projecting back through the orthonormal face axes.
    """
    mesh = request.getfixturevalue(mesh_name)
    mesh.faces_loads(LOAD)
    structure, xyz = _structure_xyz(mesh)

    faces = structure.faces_indexed
    faces_load = jnp.asarray(mesh.faces_loads())
    local_load = calculate_faces_load(xyz, faces, faces_load, is_local=True)

    target = jnp.array(LOAD)
    for index, face in enumerate(faces):
        fxyz = xyz[face, :]
        lcs = polygon_lcs(fxyz)

        # The orthonormal frame round-trips the local load back to the input,
        # and the normal projection equals the input normal (z) component.
        assert jnp.allclose(local_load[index] @ lcs.T, target, atol=1e-6)
        normal = normalize_vector(normal_polygon(fxyz, unitized=False))
        assert jnp.allclose(jnp.dot(local_load[index], normal), LOAD[2], atol=1e-6)


# ==============================================================================
# COMPAS cross-check (deletable scaffolding)
# ==============================================================================


@pytest.mark.compas_xcheck
@pytest.mark.parametrize("mesh_name", MESHES)
def test_faces_load_matches_compas_vertex_area(mesh_name, request):
    """
    The per-node global face load matches the COMPAS vertex tributary area times pz.
    """
    mesh = request.getfixturevalue(mesh_name)
    mesh.faces_loads([0.0, 0.0, PZ])
    structure, xyz = _structure_xyz(mesh)

    faces_load = jnp.asarray(mesh.faces_loads())
    nodes_load = nodes_load_from_faces(xyz, faces_load, structure)

    compas_nodes_load = []
    for vertex in structure.vertices:
        vload = mesh.vertex_area(int(vertex)) * PZ
        compas_nodes_load.append([0.0, 0.0, vload])

    assert jnp.allclose(nodes_load, jnp.array(compas_nodes_load))


@pytest.mark.compas_xcheck
@pytest.mark.parametrize("mesh_name", MESHES)
def test_local_faces_load_normal_matches_compas(mesh_name, request):
    """
    The normal component of a local face load equals the input normal load
    projected onto the COMPAS vertex normal.
    """
    mesh = request.getfixturevalue(mesh_name)
    mesh.faces_loads([0.0, 0.0, PZ])
    structure, xyz = _structure_xyz(mesh)

    faces_load = jnp.asarray(mesh.faces_loads())
    nodes_load = nodes_load_from_faces(xyz, faces_load, structure, is_local=True)

    for index, vertex in enumerate(structure.vertices):
        load = nodes_load[index]
        if float(jnp.linalg.norm(load)) < 1e-9:
            continue
        normal = jnp.array(mesh.vertex_normal(int(vertex)))
        assert jnp.allclose(jnp.dot(load, normal), jnp.linalg.norm(load), atol=1e-6)
