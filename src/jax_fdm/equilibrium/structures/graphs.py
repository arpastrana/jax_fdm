import numpy as np
from scipy.sparse import coo_matrix

import jax
import jax.numpy as jnp

import equinox as eqx

from compas.numerical import connectivity_matrix
from compas.utilities import pairwise

from jax.experimental.sparse import BCOO

from jax_fdm import DTYPE_NP
from jax_fdm import DTYPE_JAX

from jax_fdm.equilibrium.structures.mixins import IndexingMixins


# ==========================================================================
# Graph
# ==========================================================================

class Graph(eqx.Module, IndexingMixins):
    """
    A graph.
    """
    nodes: np.ndarray
    edges: np.ndarray
    edges_indexed: jax.Array
    connectivity: jax.Array
    adjacency: jax.Array

    def __init__(self, nodes, edges):
        self.nodes = nodes

        assert edges.shape[1] == 2, "Edges in graph must connect exactly 2 nodes"
        self.edges = edges
        self.edges_indexed = self._edges_indexed()

        self.connectivity = self._connectivity_matrix()
        self.adjacency = self._adjacency_matrix()

    @property
    def num_nodes(self):
        """
        The number of nodes.
        """
        return self.nodes.size

    @property
    def num_edges(self):
        """
        The number of edges.
        """
        return self.edges.shape[0]

    def _connectivity_matrix(self):
        """
        The connectivity matrix between edges and nodes.
        """
        edges_indexed = self.edges_indexed

        return jnp.asarray(connectivity_matrix(edges_indexed, "array"), dtype=DTYPE_JAX)

    def _adjacency_matrix(self):
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
    def _connectivity_matrix(self):
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
    def connectivity_scipy(self):
        """
        The connectivity matrix between edges and nodes in SciPy CSC format.
        """
        # TODO: Refactor GraphSparse to return a JAX sparse matrix instead
        edges_indexed = self.edges_indexed

        return connectivity_matrix(edges_indexed, "csc")

    def _adjacency_matrix(self):
        """
        The adjacency matrix between nodes and nodes.
        """
        edges_indexed = self.edges_indexed

        A = adjacency_matrix(edges_indexed, "coo")

        return BCOO.from_scipy_sparse(A).todense()


# ==========================================================================
# Helper functions
# ==========================================================================

def adjacency_matrix(edges, rtype="array"):
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
    return _return_matrix(A.asfptype(), rtype)


def _return_matrix(M, rtype):
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


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    num_nodes = 5
    nodes = list(range(num_nodes))
    edges = [edge for edge in pairwise(nodes)]

    nodes = jnp.array(nodes, dtype=jnp.int64)
    edges = jnp.array(edges, dtype=jnp.int64)

    graph = Graph(nodes, edges)
    graph_sparse = GraphSparse(nodes, edges)

    assert jnp.allclose(graph_sparse.connectivity, graph.connectivity)

    print("All good, cowboy!")
