from functools import partial

import numpy as np
import jax.numpy as jnp

from jax import jit

from jax.experimental.sparse import CSC

from jax_fdm.equilibrium.state import EquilibriumState
from jax_fdm.equilibrium.structure import EquilibriumStructure
from jax_fdm.equilibrium.structure import EquilibriumStructureSparse

from jax_fdm.equilibrium.sparse import sparse_solve


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    The equilibrium solver.
    """
    def __init__(self, network):
        self.structure = EquilibriumStructure(network)

    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from a force density network.
        """
        return cls(network)

    @staticmethod
    def edges_vectors(xyz, connectivity):
        """
        Calculate the unnormalized edge directions.
        """
        return connectivity @ xyz

    @staticmethod
    def edges_lengths(vectors):
        """
        Compute the length of the edges.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    @staticmethod
    def edges_forces(q, lengths):
        """
        Calculate the force in the edges.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    @staticmethod
    def nodes_residuals(q, loads, vectors, connectivity):
        """
        Compute the force at the anchor supports of the structure.
        """
        return loads - connectivity.T @ (q[:, None] * vectors)

    @staticmethod
    def nodes_positions(xyz_free, xyz_fixed, indices):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def nodes_free_positions(self, q, xyz_fixed, loads):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        # shorthand
        free = self.structure.free_nodes
        c_fixed = self.structure.connectivity_fixed
        c_free = self.structure.connectivity_free

        A = c_free.T @ (q[:, None] * c_free)
        b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

        return jnp.linalg.solve(A, b)

    @partial(jit, static_argnums=0)
    def __call__(self, q, xyz_fixed, loads):
        """
        Compute an equilibrium state using the force density method.
        """
        connectivity = self.structure.connectivity  # dense jax matrix now. BCOO, better?
        indices = self.structure.freefixed_nodes  # tuple

        xyz_free = self.nodes_free_positions(q, xyz_fixed, loads)
        xyz_all = self.nodes_positions(xyz_free, xyz_fixed, indices)
        vectors = self.edges_vectors(xyz_all, connectivity)
        residuals = self.nodes_residuals(q, loads, vectors, connectivity)
        lengths = self.edges_lengths(vectors)
        forces = self.edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz_all,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                loads=loads,
                                force_densities=q)


# ==========================================================================
# Sparse equilibrium model
# ==========================================================================

class EquilibriumModelSparse(EquilibriumModel):
    """
    The equilibrium solver. Sparse.
    """
    def __init__(self, network):
        self.structure = EquilibriumStructureSparse(network)

        # Do some precomputation to be able to construct the lhs matrix through indexing
        c_free_csc = self.structure.connectivity_scipy[:, self.structure.free_nodes]
        index_array = self._get_sparse_index_array(c_free_csc)
        self.index_array = index_array

        # Indices of data corresponding to diagonal.
        # With this array we can just index directly into the CSC.data array to refer to the diagonal entries.
        self.diag_indices = self._get_sparse_diag_indices(index_array)

        # Prepare the array D st when D.T @ q we get the diagonal elements of matrix.
        self.diags = self._get_sparse_diag_data(c_free_csc)

    def nodes_free_positions(self, q, xyz_fixed, loads):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        return sparse_solve(q,  # differentiable parameters
                            xyz_fixed,
                            loads,
                            self.structure.free_nodes,  # connectivity (non-differentiable)
                            self.structure.connectivity_free,
                            self.structure.connectivity_fixed,
                            self.index_array,  # precomputed data (non-differentiable)
                            self.diag_indices,
                            self.diags)

    def _get_sparse_index_array(self, c_free_csc):
        """
        Create an index array such that the off-diagonals can index into the force density vector.

        This array is used to create the off-diagonal entries of the lhs matrix.
        """
        force_density_modified_c_free_csc = c_free_csc.copy()
        force_density_modified_c_free_csc.data *= np.take(np.arange(c_free_csc.shape[0]) + 1, c_free_csc.indices)
        index_array = -(c_free_csc.T @ force_density_modified_c_free_csc)

        # The diagonal entries should be set to 0 so that it indexes
        # into a valid entry, but will later be overwritten.
        index_array.setdiag(0)

        return index_array.astype(int)

    @staticmethod
    def _get_sparse_diag_data(c_free_csc):
        """
        The diagonal of the lhs matrix is the sum of force densities for
        each outgoing/incoming edge on the node.

        We create the `diags` matrix such that when we multiply it with the
        force density vector we get the diagonal.
        """
        diags_data = jnp.ones_like(c_free_csc.data)

        return CSC((diags_data, c_free_csc.indices, c_free_csc.indptr), shape=c_free_csc.shape)

    @staticmethod
    def _get_sparse_diag_indices(csc):
        """
        Given a CSC matrix, get indices into `data` that access diagonal elements in order.
        """
        all_indices = []
        for i in range(csc.shape[0]):
            index_range = csc.indices[csc.indptr[i]:csc.indptr[i + 1]]
            ind_loc = jnp.where(index_range == i)[0]
            all_indices.append(ind_loc + csc.indptr[i])

        return jnp.concatenate(all_indices)
