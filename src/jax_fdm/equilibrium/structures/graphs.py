import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import spmatrix

from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_JAX
from jax_fdm import DTYPE_NP

# ==========================================================================
# Graph
# ==========================================================================

class Graph(eqx.Module):
    """
    A graph.
    """
    nodes: np.ndarray
    edges: np.ndarray
    edges_indexed: jax.Array
    connectivity: jax.Array
    adjacency: jax.Array

    def __init__(self, nodes: Int[np.ndarray, "nodes"], edges: Int[np.ndarray, "edges 2"]):
        self.nodes = nodes

        assert edges.shape[1] == 2, "Edges in graph must connect exactly 2 nodes"
        self.edges = edges
        self.edges_indexed = self._edges_indexed()

        self.connectivity = self._connectivity_matrix()
        self.adjacency = self._adjacency_matrix()

    @property
    def num_nodes(self) -> int:
        """
        The number of nodes.
        """
        return self.nodes.size

    @property
    def num_edges(self) -> int:
        """
        The number of edges.
        """
        return self.edges.shape[0]

    @property
    def node_index(self) -> dict[int, int]:
        """
        A dictionary between node keys and their enumeration indices.
        """
        return {int(node): index for index, node in enumerate(self.nodes)}

    @property
    def edge_index(self) -> dict[tuple[int, int], int]:
        """
        A dictionary between edge keys and their enumeration indices.
        """
        return {(int(u), int(v)): index for index, (u, v) in enumerate(self.edges)}

    def _edges_indexed(self) -> Int[Array, "edges 2"]:
        """
        An iterable with the edges pointing to the indices of the node keys.
        """
        node_index = self.node_index

        edges_indexed = []
        for u, v in self.edges:
            edge = node_index[int(u)], node_index[int(v)]
            edges_indexed.append(edge)

        return jnp.asarray(edges_indexed, dtype=DTYPE_INT_JAX)

    def _connectivity_matrix(self) -> Float[Array, "edges nodes"]:
        """
        The connectivity matrix between edges and nodes.
        """
        edges_indexed = self.edges_indexed

        return jnp.asarray(connectivity_matrix(edges_indexed, "array"), dtype=DTYPE_JAX)

    def _adjacency_matrix(self) -> Float[Array, "nodes nodes"]:
        """
        The adjacency matrix between nodes and nodes.
        """
        edges_indexed = self.edges_indexed

        return jnp.asarray(adjacency_matrix(edges_indexed, "array"), dtype=DTYPE_JAX)


# ==========================================================================
# Graph sparse
# ==========================================================================

class GraphSparse(Graph):
    """
    A sparse graph.
    """
    def _connectivity_matrix(self) -> Float[Array, "edges nodes"]:
        """
        The connectivity matrix between edges and nodes in JAX format.

        Notes
        -----
        This currently is a dense array, but it should be a sparse one.

        How come?

        Currently there is a JAX bug that prevents us from using the
        sparse format with the connectivity matrix:

          C =  BCOO.from_scipy_sparse(self.connectivity_scipy)

        When not using a dense array from the next line, we get the
        following error:

          TypeError: float() argument must be a string or a number, not 'Zero'

        Therefore, we use the connectivity matrix method from the parent
        class, which outputs a dense array.

        However, submatrices connectivity_free and connectivity_fixed
        are correctly initialized and used as sparse matrices.
        """
        # C = super()._connectivity_matrix()
        # return BCOO.fromdense(C).astype(DTYPE_JAX)

        # C = self.connectivity_scipy
        # return BCOO.from_scipy_sparse(C)[:, :]

        # C = self.connectivity_scipy
        # args = (C.data, C.indices, C.indptr)
        # return CSC(args, shape=C.shape)

        return super()._connectivity_matrix()

    @property
    def connectivity_scipy(self) -> csc_matrix:
        """
        The connectivity matrix between edges and nodes in SciPy CSC format.
        """
        # TODO: Refactor GraphSparse to return a JAX sparse matrix instead
        edges_indexed = self.edges_indexed

        return connectivity_matrix(edges_indexed, "csc")  # pyright: ignore[reportReturnType]  # rtype="csc" always yields a csc_matrix; connectivity_matrix's return type is a broader union across all rtype literals

    def _adjacency_matrix(self) -> Float[Array, "nodes nodes"]:
        """
        The adjacency matrix between nodes and nodes.
        """
        edges_indexed = self.edges_indexed

        A = adjacency_matrix(edges_indexed, "coo")

        return BCOO.from_scipy_sparse(A).todense()


# ==========================================================================
# Helper functions
# ==========================================================================

def connectivity_matrix(edges: Int[Array, "edges 2"], rtype: str = "array") -> np.ndarray | list | spmatrix:
    """
    Creates a connectivity matrix from a list of vertex index pairs.

    Each row represents an edge, with -1 in the start node's column and +1 in
    the end node's column.
    """
    m = len(edges)
    data = np.array([-1] * m + [1] * m)
    rows = np.array(list(range(m)) + list(range(m)))
    cols = np.array([edge[0] for edge in edges] + [edge[1] for edge in edges])

    C = coo_matrix((data, (rows, cols))).asfptype()

    return build_matrix(C, rtype)


def adjacency_matrix(edges: Int[Array, "edges 2"], rtype: str = "array") -> np.ndarray | list | spmatrix:
    """
    Creates a vertex-vertex adjacency matrix.

    It expects that vertices / nodes are continuously indexed (no skips),
    and that edges are indexed from 0 to len(vertices) / len(nodes).
    """
    num_vertices = np.max(np.ravel(edges)) + 1

    # rows and columns indices for the COO format
    rows = np.hstack([edges[:, 0], edges[:, 1]])  # add edges in both directions for undirected graph
    cols = np.hstack([edges[:, 1], edges[:, 0]])

    # data to fill in (all 1s for the existence of edges)
    data = np.ones(len(rows), dtype=DTYPE_NP)

    # create the COO matrix
    A = coo_matrix(
        (data, (rows, cols)),
        shape=(num_vertices, num_vertices)
    )

    # convert to floating point matrix
    return build_matrix(A.asfptype(), rtype)


def build_matrix(M: spmatrix, rtype: str) -> np.ndarray | list | spmatrix:
    """
    Returns a scipy sparse matrix in the requested format.
    """
    if rtype == "list":
        return M.toarray().tolist()
    if rtype == "array":
        return M.toarray()
    if rtype == "csr":
        return M.tocsr()
    if rtype == "csc":
        return M.tocsc()
    if rtype == "coo":
        return M.tocoo()
    return M
