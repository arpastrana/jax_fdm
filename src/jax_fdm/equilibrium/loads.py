import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium.structures import EquilibriumMeshStructure
from jax_fdm.equilibrium.structures import EquilibriumStructure
from jax_fdm.geometry import area_triangle
from jax_fdm.geometry import length_vector
from jax_fdm.geometry import line_lcs
from jax_fdm.geometry import normal_polygon
from jax_fdm.geometry import polygon_lcs

# ==========================================================================
# Face loads
# ==========================================================================


def nodes_load_from_faces(
    xyz: Float[Array, "nodes 3"],
    faces_load: Float[Array, "faces 3"],
    structure: EquilibriumMeshStructure,
    is_local: bool = False,
) -> Float[Array, "nodes 3"]:
    """
    Calculate the tributary face loads aplied to the nodes of a structure.
    """
    faces = structure.faces_indexed
    faces_load = calculate_faces_load(xyz, faces, faces_load, is_local)
    edges_load = edges_tributary_faces_load(xyz, faces_load, structure)
    nodes_load = nodes_tributary_edges_load(edges_load, structure)

    return nodes_load


def calculate_faces_load(
    xyz: Float[Array, "nodes 3"],
    faces: Int[Array, "faces vertices"],
    faces_load: Float[Array, "faces 3"],
    is_local: bool,
) -> Float[Array, "faces 3"]:
    """
    Transform the face loads to the XYZ cartesian coordinate system if needed.
    """
    if not is_local:
        return faces_load

    faces_load_lcs = vmap(face_load_lcs, in_axes=(None, 0, 0))
    loads = faces_load_lcs(xyz, faces, faces_load)

    return loads


def face_xyz(xyz: Float[Array, "nodes 3"], face: Int[Array, "vertices"]) -> Float[Array, "vertices 3"]:
    """
    Get this face XYZ coordinates from XYZ vertices array.
    """
    face = jnp.ravel(face)

    xyz_face = xyz[face, :]
    xyz_repl = xyz_face[0, :]

    # NOTE: Replace -1 with first entry to avoid nans in gradient computation
    # This was a pesky bug, since using nans as replacement do not cause
    # issues with the forward computation of normals, but it does for the backward pass.
    xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

    return xyz_face


def face_load_lcs(
    xyz: Float[Array, "nodes 3"],
    face: Int[Array, "vertices"],
    face_load: Float[Array, "3"],
) -> Float[Array, "3"]:
    """
    Transform the load vector applied to the face to a vector in its local coordinate system.
    """
    # fxyz = face_xyz(xyz, face)
    fxyz = xyz[face, :]

    normal = normal_polygon(fxyz)
    is_zero_normal = jnp.allclose(normal, 0.0)
    lcs = jnp.where(is_zero_normal, jnp.eye(3), polygon_lcs(fxyz))

    load = face_load @ lcs

    return load


def edges_tributary_faces_load(
    xyz: Float[Array, "nodes 3"],
    faces_load: Float[Array, "faces 3"],
    structure: EquilibriumMeshStructure,
) -> Float[Array, "edges 3"]:
    """
    Calculate the face area load taken by every edge in a datastructure.
    """
    c_vertices = structure.connectivity
    c_edges_faces = structure.connectivity_edges_faces
    c_faces_vertices = structure.connectivity_faces_vertices

    face_centroids = c_faces_vertices @ xyz
    loads_fn = vmap(edge_tributary_faces_load, in_axes=(0, 0, None, None, None))

    return loads_fn(c_vertices, c_edges_faces, xyz, faces_load, face_centroids)


def edge_tributary_faces_load(
    c_edge_nodes: Float[Array, "nodes"],
    c_edge_faces: Float[Array, "faces"],
    xyz: Float[Array, "nodes 3"],
    faces_load: Float[Array, "faces 3"],
    face_centroids: Float[Array, "faces 3"],
) -> Float[Array, "3"]:
    """
    Calculate the face area load taken by one edge in a datastructure.
    """
    indices = jnp.flatnonzero(c_edge_nodes, size=2)
    line = xyz[indices, :]

    findices = jnp.flatnonzero(c_edge_faces, size=2, fill_value=-1)
    floads = faces_load[findices, :]

    centroids = face_centroids[findices, :]

    # NOTE: correct loads if negative face indices exist (e.g. faces on boundary)
    areas = vmap(edge_tributary_face_area, in_axes=(None, 0))(line, centroids)
    areas = jnp.where(findices >= 0, areas.ravel(), 0.0)

    return areas @ floads


def edge_tributary_face_area(line: Float[Array, "2 3"], centroid: Float[Array, "3"]) -> Float[Array, ""]:
    """
    The triangle-based, face tributary area of an edge.
    """
    triangle = jnp.vstack((line, jnp.reshape(centroid, (1, 3))))

    return area_triangle(triangle)


# ==========================================================================
# Edge loads
# ==========================================================================


def nodes_load_from_edges(
    xyz: Float[Array, "nodes 3"],
    edges_load: Float[Array, "edges 3"],
    structure: EquilibriumStructure,
    is_local: bool = False,
) -> Float[Array, "nodes 3"]:
    """
    Calculate the tributary edge loads aplied to the nodes of a structure.
    """
    edges = structure.edges_indexed
    edges_load = calculate_edges_load(xyz, edges, edges_load, is_local)
    edges_load = edges_tributary_edges_load(xyz, edges_load, structure)

    return nodes_tributary_edges_load(edges_load, structure)


def calculate_edges_load(
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "edges 2"],
    edges_load: Float[Array, "edges 3"],
    is_local: bool,
) -> Float[Array, "edges 3"]:
    """
    Transform the edges load to the XYZ cartesian coordinate system if needed.
    """
    if not is_local:
        return edges_load

    edges_load_lcs = vmap(edge_load_lcs, in_axes=(None, 0, 0))

    return edges_load_lcs(xyz, edges, edges_load)


def edge_load_lcs(
    xyz: Float[Array, "nodes 3"],
    edge: Int[Array, "2"],
    edge_load: Float[Array, "3"],
) -> Float[Array, "3"]:
    """
    Transform the load vector applied to an edge to a vector in its local coordinate system.
    """
    def edge_xyz(xyz, edge):
        return xyz[edge, :]

    exyz = edge_xyz(xyz, edge)
    lcs = line_lcs(exyz)

    return edge_load @ lcs


def edges_tributary_edges_load(
    xyz: Float[Array, "nodes 3"],
    edges_load: Float[Array, "edges 3"],
    structure: EquilibriumStructure,
) -> Float[Array, "edges 3"]:
    """
    Calculate the face area load taken by every edge in a datastructure.
    """
    # TODO: edge lengths calculated by the FDM equilibrium model, inject?
    edges_vector = structure.connectivity @ xyz
    edges_length = length_vector(edges_vector)

    # scale edge load by edge length
    return edges_load * edges_length


# ==========================================================================
# Node helpers
# ==========================================================================

def nodes_tributary_edges_load(
    edges_load: Float[Array, "edges 3"],
    structure: EquilibriumStructure,
) -> Float[Array, "nodes 3"]:
    """
    Calculate the load vector applied to the nodes based on the edge loads.
    """
    return 0.5 * jnp.abs(structure.connectivity).T @ edges_load
