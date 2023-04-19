from functools import partial

import jax.numpy as jnp
import numpy as np

from jax_fdm.equilibrium.state import EquilibriumState
from jax_fdm.equilibrium.structure import EquilibriumStructure
from jax_fdm.equilibrium.sparse_solver import linear_solve, get_sparse_diag_indices
from jax.experimental.sparse import BCOO, CSC

import jax
from jax import jit
from jax import vmap

# ==========================================================================
# Equilibrium model
# ==========================================================================


class EquilibriumModel:
    """
    The equilibrium solver.
    """
    def __init__(self, network):
        self.structure = EquilibriumStructure(network)

        # Initialize the connectivity matrix

        # Currently there is a JAX bug that prevents us from using the sparse format with the connectivity matrix.
        # When `todense()` is removed from the next line, we get the following error:
        # TypeError: Value Zero(ShapedArray(float64[193,3])) with type <class 'jax._src.ad_util.Zero'> is not a valid JAX type
        self.connectivity = CSC((self.structure.connectivity.data,
                                 self.structure.connectivity.indices,
                                 self.structure.connectivity.indptr),
                                shape=self.structure.connectivity.shape).todense()
        self.c_fixed = BCOO.from_scipy_sparse(self.structure.connectivity_fixed)
        self.c_free = BCOO.from_scipy_sparse(self.structure.connectivity_free)
        self.free = self.structure.free_nodes
        self.indices = self.structure.freefixed_nodes

        ###################################################################################
        # Do some precomputation to be able to construct the lhs matrix through indexing. #
        ###################################################################################

        # Create an index array such that the off-diagonals can index into the force density vector
        # to create the off-diagonal entries of the lhs matrix.
        c_free_csc = self.structure.connectivity_free

        force_density_modified_c_free_csc = c_free_csc.copy()
        force_density_modified_c_free_csc.data *= np.take(np.arange(c_free_csc.shape[0]) + 1, c_free_csc.indices)
        index_array = -(c_free_csc.T @ force_density_modified_c_free_csc)
        # The diagonal entries should be set to 0 so that it indexes into a valid entry, but will later be overwritten.
        index_array.setdiag(0)
        self.index_array = index_array.astype(int)

        # The diagonal of the lhs matrix is the sum of force densities for each outgoing/incoming edge on the node.
        # We create the `diags` matrix such that when we multiply it with the force density vector we get the diagonal.

        # Indices of data corresponding to diagonal. With this array we can just index directly into the CSC.data
        # array to refer to the diagonal entries.
        self.diag_indices = get_sparse_diag_indices(index_array)

        # Prepare the array D st when D.T @ q we get the diagonal elements of matrix.
        diags_data = jnp.ones_like(c_free_csc.data)
        self.diags = CSC((diags_data, c_free_csc.indices, c_free_csc.indptr), shape=c_free_csc.shape)

    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from a force density network.
        """
        return cls(network)

    def edges_vectors(self, xyz):
        """
        Calculate the unnormalized edge directions.
        """
        return self.connectivity @ xyz

    def edges_lengths(self, vectors):
        """
        Compute the length of the edges.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    def edges_forces(self, q, lengths):
        """
        Calculate the force in the edges.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    def nodes_residuals(self, q, loads, vectors):
        """
        Compute the force at the anchor supports of the structure.
        """
        return loads - self.connectivity.T @ (q[:, None] * vectors)

    def nodes_free_positions(self, q, xyz_fixed, loads, sparsesolve=False):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        if sparsesolve:
            return linear_solve(q, xyz_fixed, loads,  # differentiable parameters
                                self.free, self.c_free, self.c_fixed,  # connectivity (non-differentiable)
                                self.index_array, self.diag_indices, self.diags)  # precomputed data (non-differentiable)
        else:
            c_free = self.c_free.todense()
            c_fixed = self.c_fixed.todense()

            A = c_free.T @ (q[:, None] * c_free)
            b = loads[self.free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

            return jnp.linalg.solve(A, b)

    def nodes_positions(self, xyz_free, xyz_fixed):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        # NOTE: free fixed indices sorted by enumeration
        indices = self.structure.freefixed_nodes
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    @partial(jit, static_argnums=(0, 4))
    def __call__(self, q, xyz_fixed, loads, sparsesolve=False):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz_free = self.nodes_free_positions(q, xyz_fixed, loads, sparsesolve=sparsesolve)
        xyz_all = self.nodes_positions(xyz_free, xyz_fixed)
        vectors = self.edges_vectors(xyz_all)
        residuals = self.nodes_residuals(q, loads, vectors)
        lengths = self.edges_lengths(vectors)
        forces = self.edges_forces(q, lengths)
        return EquilibriumState(xyz=xyz_all,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                loads=loads,
                                force_densities=q)
