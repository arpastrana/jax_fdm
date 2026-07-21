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

__all__ = [
    "calculate_edges_load",
    "calculate_faces_load",
    "edge_load_lcs",
    "edge_tributary_face_area",
    "edge_tributary_faces_load",
    "edges_tributary_edges_load",
    "edges_tributary_faces_load",
    "face_load_lcs",
    "face_xyz",
    "nodes_load_from_edges",
    "nodes_load_from_faces",
    "nodes_tributary_edges_load",
]


def nodes_load_from_faces(
    xyz: Float[Array, "nodes 3"],
    faces_load: Float[Array, "faces 3"],
    structure: EquilibriumMeshStructure,
    is_local: bool = False,
) -> Float[Array, "nodes 3"]:
    """
    Distribute face area loads to the nodes of a structure.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    faces_load :
        The area load on each face.
    structure :
        The mesh structure that maps faces to their edges and nodes.
    is_local :
        If True, the face load is given in the face's local coordinate system and
        is transformed to global coordinates first.

    Returns
    -------
    loads :
        The tributary face load carried by each node.

    Notes
    -----
    The face load flows to nodes in two steps: it is split among the face edges by
    triangular tributary areas, then split from edges to their nodes.
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
    Transform face loads to global cartesian coordinates when they are local.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    faces :
        The vertex indices of each face.
    faces_load :
        The load on each face.
    is_local :
        If True, the loads are in each face's local coordinate system and are
        rotated to global coordinates; if False, they are returned unchanged.

    Returns
    -------
    loads :
        The face loads expressed in global cartesian coordinates.
    """
    if not is_local:
        return faces_load

    faces_load_lcs = vmap(face_load_lcs, in_axes=(None, 0, 0))
    loads = faces_load_lcs(xyz, faces, faces_load)

    return loads


def face_xyz(
    xyz: Float[Array, "nodes 3"],
    face: Int[Array, "vertices"],
) -> Float[Array, "vertices 3"]:
    """
    Gather the coordinates of a face's vertices, padding safely for gradients.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    face :
        The vertex indices of the face, with ``-1`` padding for absent vertices.

    Returns
    -------
    xyz_face :
        The coordinates of the face's vertices.

    Notes
    -----
    Padding indices (``-1``) are replaced by the first vertex rather than left to
    index nan. nan padding is harmless in the forward normal computation but
    produces nan gradients in the backward pass.
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
    Rotate a face's local load vector into global cartesian coordinates.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    face :
        The vertex indices of the face.
    face_load :
        The load on the face, in the face's local coordinate system.

    Returns
    -------
    load :
        The face load expressed in global cartesian coordinates.

    Notes
    -----
    A degenerate face with a near-zero normal falls back to the identity frame, so
    its local load passes through unrotated instead of producing nans.
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
    Split the face area loads onto the edges of a structure.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    faces_load :
        The area load on each face, in global coordinates.
    structure :
        The mesh structure that provides node, edge-face, and face-vertex
        connectivity.

    Returns
    -------
    edges_load :
        The face load carried by each edge.
    """
    edges = structure.edges_indexed
    edges_faces = structure.edges_faces_indexed
    c_faces_vertices = structure.connectivity_faces_vertices

    face_centroids = c_faces_vertices @ xyz
    loads_fn = vmap(edge_tributary_faces_load, in_axes=(0, 0, None, None, None))

    return loads_fn(edges, edges_faces, xyz, faces_load, face_centroids)


def edge_tributary_faces_load(
    edge: Int[Array, "2"],
    edge_faces: Int[Array, "2"],
    xyz: Float[Array, "nodes 3"],
    faces_load: Float[Array, "faces 3"],
    face_centroids: Float[Array, "faces 3"],
) -> Float[Array, "3"]:
    """
    Compute the face area load taken by a single edge.

    Parameters
    ----------
    edge :
        The node index pair of the edge.
    edge_faces :
        The indices of the edge's incident faces, padded with ``-1``.
    xyz :
        The coordinates of all nodes.
    faces_load :
        The area load on each face, in global coordinates.
    face_centroids :
        The centroid of each face.

    Returns
    -------
    load :
        The face load carried by the edge.

    Notes
    -----
    The edge draws load from its (up to two) incident faces, weighted by the
    triangular area spanned by the edge and each face centroid. Padding face
    indices (``-1``) on boundary edges contribute zero.
    """
    line = xyz[edge, :]

    findices = edge_faces
    floads = faces_load[findices, :]

    centroids = face_centroids[findices, :]

    # NOTE: correct loads if negative face indices exist (e.g. faces on boundary)
    areas = vmap(edge_tributary_face_area, in_axes=(None, 0))(line, centroids)
    areas = jnp.where(findices >= 0, areas.ravel(), 0.0)

    return areas @ floads


def edge_tributary_face_area(
    line: Float[Array, "2 3"],
    centroid: Float[Array, "3"],
) -> Float[Array, ""]:
    """
    Compute an edge's tributary area within one face as a triangle area.

    Parameters
    ----------
    line :
        The two endpoint coordinates of the edge.
    centroid :
        The centroid of the incident face.

    Returns
    -------
    area :
        The area of the triangle formed by the edge and the face centroid.
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
    Distribute edge line loads to the nodes of a structure.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    edges_load :
        The line load on each edge.
    structure :
        The structure that maps edges to their nodes.
    is_local :
        If True, the edge load is given in the edge's local coordinate system and
        is transformed to global coordinates first.

    Returns
    -------
    loads :
        The tributary edge load carried by each node.

    Notes
    -----
    The line load is scaled by edge length to a total force, then split equally
    between the edge's two nodes.
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
    Transform edge loads to global cartesian coordinates when they are local.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    edges :
        The node indices of each edge.
    edges_load :
        The load on each edge.
    is_local :
        If True, the loads are in each edge's local coordinate system and are
        rotated to global coordinates; if False, they are returned unchanged.

    Returns
    -------
    loads :
        The edge loads expressed in global cartesian coordinates.
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
    Rotate an edge's local load vector into global cartesian coordinates.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    edge :
        The two node indices of the edge.
    edge_load :
        The load on the edge, in the edge's local coordinate system.

    Returns
    -------
    load :
        The edge load expressed in global cartesian coordinates.
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
    Scale per-length edge loads into total edge forces by edge length.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    edges_load :
        The line load on each edge, per unit length.
    structure :
        The structure that provides the node connectivity matrix.

    Returns
    -------
    edges_load :
        The total load on each edge, the per-length load times the edge length.
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
    Split each edge's total load equally between its two nodes.

    Parameters
    ----------
    edges_load :
        The total load on each edge.
    structure :
        The structure that provides the node connectivity matrix.

    Returns
    -------
    loads :
        The load carried by each node, half of each incident edge's load.

    Notes
    -----
    Scatter-adds half of each edge's load onto both of its endpoints, which is
    equivalent to ``0.5 * |C|.T @ edges_load`` but does not require the
    connectivity matrix, whose absolute value sparse arrays cannot take.
    """
    edges = structure.edges_indexed
    loads = jnp.zeros((structure.num_nodes, 3))
    loads = loads.at[edges[:, 0]].add(0.5 * edges_load)
    loads = loads.at[edges[:, 1]].add(0.5 * edges_load)

    return loads
