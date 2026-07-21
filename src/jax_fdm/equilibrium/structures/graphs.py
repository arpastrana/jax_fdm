import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_JAX
from jax_fdm import DTYPE_NP

# ==========================================================================
# Graph
# ==========================================================================

__all__ = [
    "Graph",
    "GraphSparse",
    "adjacency_matrix",
    "connectivity_matrix",
]


class Graph(eqx.Module):
    """
    An immutable graph holding node and edge connectivity as dense matrices.

    Notes
    -----
    An equinox Module, so instances are registered pytrees. Nodes and edges are
    kept as static NumPy index arrays, while the connectivity and adjacency
    matrices are JAX arrays used in the differentiable equilibrium computation.
    """

    nodes: Int[np.ndarray, "nodes"]
    edges: Int[np.ndarray, "edges 2"]
    edges_indexed: Int[Array, "edges 2"]
    connectivity: Float[Array, "edges nodes"]
    adjacency: Float[Array, "nodes nodes"]

    def __init__(
        self,
        nodes: Int[np.ndarray, "nodes"],
        edges: Int[np.ndarray, "edges 2"],
    ) -> None:
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
        The edges rewritten from node keys to contiguous node indices.
        """
        node_index = self.node_index

        edges_indexed = []
        for u, v in self.edges:
            edge = node_index[int(u)], node_index[int(v)]
            edges_indexed.append(edge)

        return jnp.asarray(edges_indexed, dtype=DTYPE_INT_JAX)

    def _connectivity_matrix(self) -> Float[Array, "edges nodes"]:
        """
        The signed edge-node incidence matrix of the graph.
        """
        C = connectivity_matrix(self.edges_indexed, self.num_nodes).toarray()

        return jnp.asarray(C, dtype=DTYPE_JAX)

    def _adjacency_matrix(self) -> Float[Array, "nodes nodes"]:
        """
        The symmetric node-node adjacency matrix of the graph.
        """
        edges_indexed = self.edges_indexed

        return jnp.asarray(
            adjacency_matrix(edges_indexed, self.num_nodes).toarray(),
            dtype=DTYPE_JAX,
        )


# ==========================================================================
# Graph sparse
# ==========================================================================


class GraphSparse(Graph):
    """
    A graph that keeps its connectivity and adjacency matrices in sparse format.
    """

    # The sparse subclass deliberately swaps the dense matrices for JAX sparse
    # arrays; the narrowed fields and builders are the point, not a slip
    connectivity: Float[BCOO, "edges nodes"]
    adjacency: Float[BCOO, "nodes nodes"]

    def _connectivity_matrix(self) -> Float[BCOO, "edges nodes"]:
        """
        The signed edge-node incidence matrix, in sparse format.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy)

    @property
    def connectivity_scipy(self) -> csc_matrix:
        """
        The signed edge-node incidence matrix as a scipy sparse matrix.

        This is the assembly substrate the sparse structures build from: the
        free and fixed column submatrices and the stiffness precomputation
        slice and factor it before converting the results to JAX sparse format.

        Notes
        -----
        Kept in scipy format on purpose. Slicing columns out of a JAX sparse
        matrix materializes a stored entry for every edge-column pair rather
        than only the nonzeros, and the stiffness precomputation relies on
        structural operations (transpose products, diagonal rewrites, index
        pointer reads) that JAX sparse arrays do not provide. Rebuilt on every
        access, which costs fractions of a millisecond.
        """
        return connectivity_matrix(self.edges_indexed, self.num_nodes)

    def _adjacency_matrix(self) -> Float[BCOO, "nodes nodes"]:
        """
        The symmetric node-node adjacency matrix, in sparse format.
        """
        edges_indexed = self.edges_indexed

        A = adjacency_matrix(edges_indexed, self.num_nodes)

        return BCOO.from_scipy_sparse(A)


# ==========================================================================
# Helper functions
# ==========================================================================


def connectivity_matrix(
    edges: Int[Array, "edges 2"],
    num_nodes: int | None = None,
) -> csc_matrix:
    """
    Build the signed edge-node incidence matrix from indexed edges.

    Parameters
    ----------
    edges :
        The node index pair of each edge.
    num_nodes :
        The total number of nodes, fixing the column count. When omitted, it is
        inferred as the largest node index plus one, which undercounts columns
        if the highest-indexed node touches no edge.

    Returns
    -------
    connectivity :
        The incidence matrix in sparse format, one row per edge, with ``-1``
        in the start node's column and ``+1`` in the end node's column.
    """
    # Iterating a JAX array element-wise costs one device sync per element;
    # convert to NumPy once and slice columns vectorized.
    edges_np = np.asarray(edges)
    m = len(edges_np)
    data = np.concatenate((-np.ones(m), np.ones(m)))
    rows = np.concatenate((np.arange(m), np.arange(m)))
    cols = np.concatenate((edges_np[:, 0], edges_np[:, 1]))

    n = num_nodes if num_nodes is not None else int(np.max(edges_np)) + 1
    shape = (m, n)

    # coo_matrix.tocsc() yields a csc_matrix at runtime; scipy's bundled stubs
    # widen the return to csc_array
    return coo_matrix((data, (rows, cols)), shape=shape).tocsc()  # pyright: ignore[reportReturnType]


def adjacency_matrix(
    edges: Int[Array, "edges 2"],
    num_nodes: int | None = None,
) -> csc_matrix:
    """
    Build the symmetric node-node adjacency matrix from indexed edges.

    Parameters
    ----------
    edges :
        The node index pair of each edge.
    num_nodes :
        The total number of nodes, fixing the matrix size. When omitted, it is
        inferred as the largest node index plus one, which undersizes the matrix
        if the highest-indexed node touches no edge.

    Returns
    -------
    adjacency :
        The symmetric adjacency matrix in sparse format, with a one wherever
        two nodes share an edge.
    """
    edges_np = np.asarray(edges)
    n = num_nodes if num_nodes is not None else int(np.max(np.ravel(edges_np))) + 1

    # add edges in both directions for undirected graph
    rows = np.concatenate((edges_np[:, 0], edges_np[:, 1]))
    cols = np.concatenate((edges_np[:, 1], edges_np[:, 0]))

    # data to fill in (all 1s for the existence of edges)
    data = np.ones(len(rows), dtype=DTYPE_NP)

    A = coo_matrix((data, (rows, cols)), shape=(n, n))

    # coo_matrix.tocsc() yields a csc_matrix at runtime; scipy's bundled stubs
    # widen the return to csc_array
    return A.tocsc()  # pyright: ignore[reportReturnType]
