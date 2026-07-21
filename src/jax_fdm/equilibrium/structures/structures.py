import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import BCSR
from jax.experimental.sparse import CSC
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.sparse import csc_matrix

from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_INT_NP
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium.structures.graphs import Graph
from jax_fdm.equilibrium.structures.graphs import GraphSparse
from jax_fdm.equilibrium.structures.meshes import Mesh
from jax_fdm.equilibrium.structures.meshes import MeshSparse

# ==========================================================================
# Structure
# ==========================================================================

__all__ = [
    "EquilibriumMeshStructure",
    "EquilibriumMeshStructureSparse",
    "EquilibriumStructure",
    "EquilibriumStructureSparse",
]


class EquilibriumStructure(Graph):
    """
    A graph with supports and the connectivity matrices the FDM solve needs.

    Notes
    -----
    Extends [Graph][jax_fdm.equilibrium.structures.graphs.Graph] with a support
    mask and the free/fixed node partition, precomputing the free and fixed
    connectivity submatrices and the index maps that reorder nodes between the
    free-fixed and native orderings.
    """

    supports: Int[np.ndarray, "nodes"]

    connectivity_free: Float[Array, "edges nodes_free"]
    connectivity_fixed: Float[Array, "edges nodes_fixed"]

    indices_free: Int[Array, "nodes_free"]
    indices_fixed: Int[Array, "nodes_fixed"]
    indices_freefixed: Int[Array, "nodes"]

    def __init__(
        self,
        nodes: Int[np.ndarray, "nodes"],
        edges: Int[np.ndarray, "edges 2"],
        supports: Int[np.ndarray, "nodes"],
        **kwargs,
    ) -> None:
        super().__init__(nodes=nodes, edges=edges, **kwargs)

        self.supports = supports

        self.indices_free = self._indices_free()
        self.indices_fixed = self._indices_fixed()
        self.indices_freefixed = self._indices_freefixed()

        self.connectivity_free = self._connectivity_free()
        self.connectivity_fixed = self._connectivity_fixed()

    @classmethod
    def from_network(cls, network: FDNetwork) -> "EquilibriumStructure":
        """
        Build an equilibrium structure from a force density network.

        Parameters
        ----------
        network :
            The network to read nodes, edges, and supports from.

        Returns
        -------
        structure :
            The structure with the network's connectivity and support mask.
        """
        nodes = list(network.nodes())
        edges = list(network.edges())

        supports = []
        for node in nodes:
            flag = 0.0
            # network.nodes() overload resolution picks the (key, attrs) tuple
            # form since data defaults to False in the untyped COMPAS stub; node
            # is always a bare int key here
            if network.is_node_support(node):
                flag = 1.0
            supports.append(flag)

        nodes = np.asarray(nodes, dtype=DTYPE_INT_NP)
        edges = np.asarray(edges, dtype=DTYPE_INT_NP)
        supports = np.asarray(supports, dtype=DTYPE_INT_NP)

        return cls(nodes, edges, supports)

    @property
    def num_supports(self) -> Int[Array, ""]:
        """
        The number of supports.
        """
        return jnp.count_nonzero(self.supports)

    @property
    def num_free(self) -> Int[Array, ""]:
        """
        The number of free (unsupported) nodes.
        """
        return self.num_nodes - self.num_supports

    @property
    def support_index(self) -> dict[int, int]:
        """
        A mapping from support keys to indices.
        """
        return {int(key): index for index, key in enumerate(self.nodes_fixed)}

    @property
    def nodes_free(self) -> Int[np.ndarray, "nodes_free"]:
        """
        The free nodes.
        """
        return self.nodes[self.indices_free]

    @property
    def nodes_fixed(self) -> Int[np.ndarray, "nodes_fixed"]:
        """
        The fixed nodes.
        """
        return self.nodes[self.indices_fixed]

    def _connectivity_free(self) -> Float[Array, "edges nodes_free"]:
        """
        The edge-node connectivity matrix restricted to the free nodes.
        """
        return self.connectivity[:, self.indices_free]

    def _connectivity_fixed(self) -> Float[Array, "edges nodes_fixed"]:
        """
        The edge-node connectivity matrix restricted to the fixed nodes.
        """
        return self.connectivity[:, self.indices_fixed]

    def _indices_free(self) -> Int[Array, "nodes_free"]:
        """
        The indices of the free (unsupported) nodes in the structure.
        """
        # jnp.flatnonzero's size kwarg must be a static int; num_free is a
        # concrete 0-d array here (supports is a static NumPy array), so int()
        # is eager and exact.
        indices = jnp.flatnonzero(self.supports == 0, size=int(self.num_free))

        return indices

    def _indices_fixed(self) -> Int[Array, "nodes_fixed"]:
        """
        The indices of the fixed (supported) nodes in the structure.
        """
        # jnp.flatnonzero's size kwarg must be a static int; num_supports is a
        # concrete 0-d array here (supports is a static NumPy array), so int()
        # is eager and exact.
        indices = jnp.flatnonzero(self.supports, size=int(self.num_supports))

        return indices

    def _indices_freefixed(self) -> Int[Array, "nodes"]:
        """
        The position of each node within the concatenated free-then-fixed ordering.

        Notes
        -----
        Inverts the free-then-fixed permutation: indexing a free-fixed-stacked
        array by this map restores the structure's native node order.
        """
        freefixed = np.concatenate(
            [
                np.asarray(self.indices_free),
                np.asarray(self.indices_fixed),
            ],
        )

        # argsort of a permutation is its inverse
        return jnp.asarray(np.argsort(freefixed), dtype=DTYPE_INT_JAX)


# ==========================================================================
# Sparse structure
# ==========================================================================


class EquilibriumStructureSparse(EquilibriumStructure, GraphSparse):
    """
    An equilibrium structure that precomputes sparse stiffness-assembly helpers.

    Notes
    -----
    Precomputes an index array, the diagonal data positions, and a diagonal
    selector matrix so the sparse model can assemble the stiffness matrix by
    indexing rather than rebuilding its structure at every solve.
    """

    connectivity_free: Float[BCOO, "edges nodes_free"]
    connectivity_fixed: Float[BCOO, "edges nodes_fixed"]
    diag_indices: Int[Array, "nodes_free"]
    index_array: Int[CSC, "nodes_free nodes_free"]
    diags: Float[BCSR, "nodes_free edges"]

    def __init__(
        self,
        nodes: Int[np.ndarray, "nodes"],
        edges: Int[np.ndarray, "edges 2"],
        supports: Int[np.ndarray, "nodes"],
        **kwargs,
    ) -> None:
        super().__init__(nodes=nodes, edges=edges, supports=supports, **kwargs)

        # Do some precomputation to be able to construct
        # the lhs matrix through indexing
        c_free_csc = self.connectivity_scipy[:, self.indices_free]
        index_array = self._get_sparse_index_array(c_free_csc)
        self.index_array = index_array

        # Indices of data corresponding to diagonal.
        # With this array we can just index directly into the
        # CSC.data array to refer to the diagonal entries.
        self.diag_indices = self._get_sparse_diag_indices(index_array)

        # Prepare the array D st when D.T @ q we get the diagonal elements of matrix
        self.diags = self._get_sparse_diag_data(c_free_csc)

    def _connectivity_free(self) -> Float[BCOO, "edges nodes_free"]:
        """
        The sparse edge-node connectivity matrix restricted to the free nodes.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.indices_free])

    def _connectivity_fixed(self) -> Float[BCOO, "edges nodes_fixed"]:
        """
        The sparse edge-node connectivity matrix restricted to the fixed nodes.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.indices_fixed])

    @staticmethod
    def _get_sparse_index_array(
        c_free_csc: csc_matrix,
    ) -> Int[CSC, "nodes_free nodes_free"]:
        """
        Build an index array mapping stiffness off-diagonals to force densities.

        Parameters
        ----------
        c_free_csc :
            The free-node connectivity matrix as a scipy sparse array.

        Returns
        -------
        index_array :
            A sparse integer array whose off-diagonal entries index into the force
            density vector when assembling the stiffness matrix.

        Notes
        -----
        The diagonal entries are set to zero so they index a valid location; the
        sparse model overwrites them with the per-node force density sums.
        """
        fd_mod_c_free_csc = c_free_csc.copy()
        # csc_matrix.shape is annotated as tuple[int, ...] | None in scipy's
        # bundled stubs, but is always populated at runtime
        fd_mod_c_free_csc.data *= np.take(
            np.arange(c_free_csc.shape[0]) + 1,  # pyright: ignore[reportOptionalSubscript]
            c_free_csc.indices,
        )
        index_array = -(c_free_csc.T @ fd_mod_c_free_csc)

        # The diagonal entries should be set to 0 so that it indexes
        # into a valid entry, but will later be overwritten.
        index_array.setdiag(0)
        index_array = index_array.astype(int)

        return CSC(
            (index_array.data, index_array.indices, index_array.indptr),
            shape=index_array.shape,
        )

    @staticmethod
    def _get_sparse_diag_indices(csc: CSC) -> Int[Array, "nodes_free"]:
        """
        Locate the diagonal entries within a sparse matrix's data array.

        Parameters
        ----------
        csc :
            The sparse matrix whose diagonal positions are sought.

        Returns
        -------
        diag_indices :
            The positions in the ``data`` array that hold the diagonal entries, in
            row order.
        """
        # Vectorized in NumPy: a per-row jnp.where loop pays one device
        # dispatch per row, which dominates the structure build time.
        indices = np.asarray(csc.indices)
        indptr = np.asarray(csc.indptr)

        # The column of each entry in the data array, in storage order.
        columns = np.repeat(np.arange(csc.shape[0]), np.diff(indptr))

        return jnp.asarray(np.flatnonzero(indices == columns), dtype=DTYPE_INT_JAX)

    @staticmethod
    def _get_sparse_diag_data(
        c_free_csc: csc_matrix,
    ) -> Float[BCSR, "nodes_free edges"]:
        """
        Build the selector matrix that maps force densities to the diagonal.

        Parameters
        ----------
        c_free_csc :
            The free-node connectivity matrix as a scipy sparse array.

        Returns
        -------
        diags :
            A sparse matrix such that its product with the force density vector
            yields the stiffness diagonal, the per-node sum of incident force
            densities.

        Notes
        -----
        Stored in sparse row format rather than column format so that Jacobians of
        the product flow through.
        """
        diags_data = jnp.ones_like(c_free_csc.data)

        args = (diags_data, c_free_csc.indices, c_free_csc.indptr)
        shape = c_free_csc.shape
        # diag_matrix = CSC(args, shape=shape)

        # NOTE: temporary change from CSV to BCSR matrix to enable Jacobians
        diag_matrix = csc_matrix(args, shape).tocsr().T
        diag_matrix = BCSR.from_scipy_sparse(diag_matrix)

        return diag_matrix


# ==========================================================================
# Mesh structures
# ==========================================================================


class EquilibriumMeshStructure(EquilibriumStructure, Mesh):
    """
    An equilibrium structure that also carries mesh face topology.

    Notes
    -----
    Adds face connectivity to
    [EquilibriumStructure][jax_fdm.equilibrium.structures.structures.EquilibriumStructure],
    so mesh vertices play
    the role of nodes while faces enable tributary face-load distribution.
    """

    def __init__(
        self,
        vertices: Int[np.ndarray, "vertices"],
        faces: Int[np.ndarray, "faces vertices"],
        edges: Int[np.ndarray, "edges 2"],
        supports: Int[np.ndarray, "vertices"],
        **kwargs,
    ) -> None:
        super().__init__(
            nodes=vertices,
            edges=edges,
            supports=supports,
            vertices=vertices,
            faces=faces,
            **kwargs,
        )

    @property
    def num_free(self) -> Int[Array, ""]:
        """
        The number of free (unsupported) vertices.
        """
        return self.num_vertices - self.num_supports

    @classmethod
    def from_mesh(cls, mesh: FDMesh) -> "EquilibriumMeshStructure":
        """
        Build an equilibrium mesh structure from a force density mesh.

        Parameters
        ----------
        mesh :
            The mesh to read vertices, faces, edges, and supports from.

        Returns
        -------
        structure :
            The mesh structure with connectivity, support mask, and face topology.

        Raises
        ------
        AssertionError
            If any face has fewer than three vertices.

        Notes
        -----
        Faces are padded with ``-1`` to a common length so they fit a rectangular
        index array; the padding is masked out downstream.
        """
        vertices = list(mesh.vertices())
        edges = list(mesh.edges())

        supports = []
        for vertex in vertices:
            flag = 0.0
            # mesh.vertices() overload resolution picks the (key, attrs) tuple
            # form since data defaults to False in the untyped COMPAS stub;
            # vertex is always a bare int key here
            if mesh.is_vertex_support(vertex):
                flag = 1.0
            supports.append(flag)

        face_keys = np.asarray(list(mesh.faces()), dtype=DTYPE_INT_NP)
        faces = [mesh.face_vertices(fkey) for fkey in face_keys]
        max_length_face = max(len(face) for face in faces)
        assert max_length_face > 2, "The mesh faces must have at least 3 vertices each"

        padded_faces = []
        pad_value = -1
        for face in faces:
            len_face = len(face)
            if len_face < max_length_face:
                face_padding = [pad_value] * (max_length_face - len_face)
                face = face + face_padding
            padded_faces.append(face)

        faces = np.asarray(padded_faces, dtype=DTYPE_INT_NP)
        vertices = np.asarray(vertices, dtype=DTYPE_INT_NP)
        edges = np.asarray(edges, dtype=DTYPE_INT_NP)
        supports = np.asarray(supports, dtype=DTYPE_INT_NP)

        return cls(vertices, faces, edges=edges, supports=supports, face_keys=face_keys)

    @property
    def support_index(self) -> dict[int, int]:
        """
        A mapping from support vertices keys to indices.
        """
        return {int(key): index for index, key in enumerate(self.vertices_fixed)}

    @property
    def vertices_free(self) -> Int[np.ndarray, "vertices_free"]:
        """
        The free (unsupported) vertices.
        """
        return self.vertices[self.indices_free]

    @property
    def vertices_fixed(self) -> Int[np.ndarray, "vertices_fixed"]:
        """
        The fixed (supported) vertices.
        """
        return self.vertices[self.indices_fixed]


class EquilibriumMeshStructureSparse(
    EquilibriumMeshStructure,
    EquilibriumStructureSparse,
    MeshSparse,
):
    """
    A mesh equilibrium structure with sparse stiffness-assembly helpers.

    Notes
    -----
    Combines the face topology of
    [EquilibriumMeshStructure][jax_fdm.equilibrium.structures.structures.EquilibriumMeshStructure]
    with the sparse assembly precomputation of
    [EquilibriumStructureSparse][jax_fdm.equilibrium.structures.structures.EquilibriumStructureSparse].
    """

    def __init__(
        self,
        vertices: Int[np.ndarray, "vertices"],
        faces: Int[np.ndarray, "faces vertices"],
        edges: Int[np.ndarray, "edges 2"],
        supports: Int[np.ndarray, "vertices"],
        **kwargs,
    ) -> None:
        super().__init__(
            vertices=vertices,
            faces=faces,
            edges=edges,
            supports=supports,
            **kwargs,
        )
