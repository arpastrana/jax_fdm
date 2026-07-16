import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.sparse import coo_matrix
from scipy.sparse import spmatrix

from compas.datastructures import Mesh as CompasMesh
from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_INT_NP
from jax_fdm.equilibrium.structures.graphs import Graph
from jax_fdm.equilibrium.structures.graphs import GraphSparse
from jax_fdm.equilibrium.structures.graphs import build_matrix

# ==========================================================================
# Mesh
# ==========================================================================


class Mesh(Graph):
    """
    A graph enriched with face topology and face-vertex, edge-face connectivity.

    Notes
    -----
    Extends :class:`Graph` with faces. Faces are stored as a rectangular index
    array padded with ``-1``, since faces may have different numbers of vertices.
    Edges are derived from the faces when not supplied and carry no topological
    meaning beyond storing per-edge data.
    """

    # The faces array can have rows of different lengths. How to handle it?
    # Using a tuple instead of an array?
    vertices: Int[np.ndarray, "vertices"]
    faces: Int[np.ndarray, "faces vertices"]
    faces_indexed: Int[Array, "faces vertices"]
    connectivity_faces_vertices: Float[Array, "faces vertices"]
    connectivity_edges_faces: Float[Array, "edges faces"]
    face_keys: Int[np.ndarray, "faces"]

    def __init__(
        self,
        vertices: Int[np.ndarray, "vertices"],
        faces: Int[np.ndarray, "faces vertices"],
        edges: Int[np.ndarray, "edges 2"] | None = None,
        face_keys: Int[np.ndarray, "faces"] | None = None,
        **kwargs,
    ):

        if edges is None:
            edges = self._edges_from_faces(faces)

        self.vertices = vertices
        assert faces.shape[1] > 2, "Mesh faces must connect at least 3 vertices"

        self.faces = faces
        self.faces_indexed = self._faces_indexed()
        self.connectivity_faces_vertices = self._connectivity_faces_matrix()

        if face_keys is None:
            face_keys = np.arange(len(faces))
        self.face_keys = np.asarray(face_keys, dtype=DTYPE_INT_NP)

        super().__init__(vertices, edges)

        self.connectivity_edges_faces = self._connectivity_edges_faces_matrix()

    @property
    def num_vertices(self) -> int:
        """
        The number of vertices.
        """
        return self.vertices.size

    @property
    def num_faces(self) -> int:
        """
        The number of faces.
        """
        return self.faces.shape[0]

    @property
    def vertex_index(self) -> dict[int, int]:
        """
        A dictionary between vertex keys and their enumeration indices.
        """
        return {int(vkey): index for index, vkey in enumerate(self.vertices)}

    @property
    def face_index(self) -> dict[int, int]:
        """
        A dictionary between face keys and their enumeration indices.
        """
        return {int(fkey): index for index, fkey in enumerate(self.face_keys)}

    def _faces_indexed(self) -> Int[Array, "faces vertices"]:
        """
        The faces rewritten from vertex keys to contiguous vertex indices.

        Notes
        -----
        Padding entries (``-1``) are preserved as-is rather than remapped, so the
        rectangular face array keeps its shape.
        """
        vertex_index = self.vertex_index

        findexed = []
        for face in self.faces:
            face_indices = []

            for vertex in face:
                u = int(vertex)
                if u >= 0:
                    index = vertex_index[u]
                else:
                    index = u
                face_indices.append(index)

            findexed.append(tuple(face_indices))

        return jnp.asarray(findexed, dtype=DTYPE_INT_JAX)

    def _connectivity_edges_faces_matrix(self) -> Float[Array, "edges faces"]:
        """
        The edge-face incidence matrix of the mesh, with a one per incident face.
        """
        C = np.zeros((self.num_edges, self.num_faces))

        for eindex, findex in enumerate(self._edges_faces()):
            C[eindex, findex] = 1.0

        return jnp.asarray(C)

    def _edges_faces(self) -> tuple[tuple[int, ...], ...]:
        """
        The face indices incident to each edge.

        Returns
        -------
        edges_faces :
            For each edge, the indices of the faces that contain it (usually one
            or two).

        Notes
        -----
        An edge shared by more than two faces is non-manifold; it is kept but a
        warning is printed, since it can distort area-load distribution.
        """
        edges_faces = []
        for u, v in self.edges:
            edge = (int(u), int(v))
            findices = []

            for findex, face in enumerate(self.faces):
                face = [vkey for vkey in face if vkey >= 0]
                face_loop = np.concatenate((face, face[:1]))
                for u, v in zip(face_loop, face_loop[1:]):
                    # iterate one one time clockwise, another counter clockwise
                    halfedge1 = (int(u), int(v))
                    halfedge2 = (int(v), int(u))

                    if edge == halfedge1 or edge == halfedge2:
                        findices.append(findex)

            # NOTE: Temporary disabled assertion
            # assert len(findices) <= 2

            if len(findices) > 2:
                print(
                    f"Warning: Edge {edge} is non-manifold, it's shared by "
                    f"({len(findices)}) faces. This might lead to unexpected "
                    f"behavior in e.g. in area load calculations.",
                )

            edges_faces.append(tuple(findices))

        return tuple(edges_faces)

    @staticmethod
    def _edges_from_faces(
        faces: Int[np.ndarray, "faces vertices"],
    ) -> Int[np.ndarray, "edges 2"]:
        """
        Derive the unique undirected edges of the mesh from its faces.

        Parameters
        ----------
        faces :
            The vertex indices of each face.

        Returns
        -------
        edges :
            The node index pair of each unique edge.

        Notes
        -----
        Every face halfedge is collected, then deduplicated so that an edge and
        its reverse count once. These edges carry no topological meaning; they
        exist only to store per-edge data.
        """
        halfedges = []
        for face in faces:
            face_loop = np.concatenate((face, face[:1]))
            for u, v in zip(face_loop, face_loop[1:]):
                halfedge = (int(u), int(v))
                halfedges.append(halfedge)

        edges = []
        visited = set()
        for u, v in halfedges:
            if (u, v) in visited or (v, u) in visited:
                continue
            edge = (u, v)
            visited.add(edge)
            edges.append(edge)

        return np.asarray(edges, dtype=DTYPE_INT_NP)

    def _connectivity_faces_matrix(self) -> Float[Array, "faces vertices"]:
        """
        The row-normalized face-vertex incidence matrix, a face centroid operator.
        """
        F = face_matrix(self.faces_indexed, "array", normalize=True)

        return jnp.asarray(F)


# ==========================================================================
# Mesh sparse
# ==========================================================================


class MeshSparse(Mesh, GraphSparse):
    """
    A mesh that assembles its connectivity through sparse intermediates.

    Notes
    -----
    The connectivity matrices are densified after a sparse build rather than kept
    sparse: leaving them sparse breaks reverse-mode gradients with
    ``TypeError: float() argument must be a string or a number, not 'Zero'``,
    raised from ``bcoo._bcoo_dot_general_transpose``.
    """

    def _connectivity_faces_matrix(self) -> Float[Array, "faces vertices"]:
        """
        The row-normalized face-vertex incidence matrix, built through sparse COO.
        """
        F = face_matrix(self.faces_indexed, "csc", normalize=True)

        return BCOO.from_scipy_sparse(F).todense()

    def _connectivity_edges_faces_matrix(self) -> Float[Array, "edges faces"]:
        """
        The edge-face incidence matrix of the mesh.
        """
        C = np.zeros((self.num_edges, self.num_faces))

        for eindex, findex in enumerate(self._edges_faces()):
            C[eindex, findex] = 1.0

        return jnp.asarray(C)


# ==========================================================================
# Helper functions
# ==========================================================================


def mesh_edges_faces(mesh: CompasMesh) -> list[tuple[int, ...]]:
    """
    List the face indices incident to each edge of a COMPAS mesh.

    Parameters
    ----------
    mesh :
        The COMPAS mesh to read edge-face incidence from.

    Returns
    -------
    edges_faces :
        For each edge, the indices of its incident faces (at most two).
    """
    face_index = {fkey: idx for idx, fkey in enumerate(mesh.faces())}

    edges_faces = []
    for u, v in mesh.edges():
        findices = []
        for fkey in mesh.edge_faces((u, v)):
            if fkey is None:
                continue

            findex = face_index[fkey]
            findices.append(findex)

        assert len(findices) <= 2

        edges_faces.append(tuple(findices))

    return edges_faces


def mesh_connectivity_edges_faces(mesh: CompasMesh) -> Float[np.ndarray, "edges faces"]:
    """
    Build the edge-face incidence matrix of a COMPAS mesh.

    Parameters
    ----------
    mesh :
        The COMPAS mesh to read edge-face incidence from.

    Returns
    -------
    connectivity :
        The incidence matrix with a one wherever an edge belongs to a face.
    """
    num_edges = len(list(mesh.edges()))
    num_faces = mesh.number_of_faces()

    connectivity = np.zeros((num_edges, num_faces))

    edges_faces = mesh_edges_faces(mesh)

    for eindex, findex in enumerate(edges_faces):
        connectivity[eindex, findex] = 1.0

    return connectivity


def face_matrix(
    face_vertices: Int[Array, "faces vertices"],
    rtype: str = "array",
    normalize: bool = True,
) -> Float[np.ndarray, "faces vertices"] | list | spmatrix:
    """
    Build a face-vertex incidence matrix, ignoring padding vertices.

    Parameters
    ----------
    face_vertices :
        The vertex indices of each face, with ``-1`` padding for absent vertices.
    rtype :
        The return format: ``"array"``, ``"list"``, ``"csr"``, ``"csc"``, or
        ``"coo"``.
    normalize :
        If True, each row sums to one, so the matrix averages vertex quantities
        into face centroids; if False, each row sums to the face's vertex count.

    Returns
    -------
    face_matrix :
        The face-vertex incidence matrix in the requested format.
    """
    face_vertices_clean = []
    for face in face_vertices:
        face_clean = [vertex for vertex in face if vertex >= 0]
        face_vertices_clean.append(face_clean)

    if normalize:
        f = np.array(
            [
                (i, j, 1.0 / len(vertices))
                for i, vertices in enumerate(face_vertices_clean)
                for j in vertices
            ],
        )
    else:
        f = np.array(
            [
                (i, j, 1.0)
                for i, vertices in enumerate(face_vertices_clean)
                for j in vertices
            ],
        )

    F = coo_matrix((f[:, 2], (f[:, 0].astype(int), f[:, 1].astype(int))))

    return build_matrix(F, rtype)
