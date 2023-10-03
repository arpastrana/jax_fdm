from functools import partial

from jax import jit
import jax.numpy as jnp

from jax.experimental.sparse import CSC

from jax_fdm.equilibrium.state import EquilibriumState
from jax_fdm.equilibrium.sparse import sparse_solve


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    The equilibrium model.
    """
    # ----------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from a force density network.

        This method exists primarily for compatibility with older versions of the library.
        Otherwise, it just returns a model that is unaware of the input network.
        """
        return cls()

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    @staticmethod
    def edges_vectors(xyz, connectivity):
        """
        Calculate the unnormalized edge directions (nodal coordinate differences).
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

    # ----------------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------------

    @staticmethod
    def nodes_residuals(q, loads, vectors, connectivity):
        """
        Compute the residual forces on the nodes of the structure.
        """
        return loads - connectivity.T @ (q[:, None] * vectors)

    @staticmethod
    def nodes_positions(xyz_free, xyz_fixed, indices):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def nodes_free_positions(self, q, xyz_fixed, loads, structure):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        A = self.stiffness_matrix(q, structure)
        b = self.force_matrix(q, xyz_fixed, loads, structure)

        return jnp.linalg.solve(A, b)

    # --------------------------------------------------------------------
    #  Call me, maybe
    # --------------------------------------------------------------------

    @partial(jit, static_argnums=(0, ))
    def __call__(self, params, structure):
        """
        Compute an equilibrium state using the force density method.
        """
        q, xyz_fixed, loads = params
        indices = structure.indices_freefixed

        xyz_free = self.nodes_free_positions(q, xyz_fixed, loads, structure)
        xyz = self.nodes_positions(xyz_free, xyz_fixed, indices)

        return self.equilibrium_state(q, xyz, loads, structure)

    # ------------------------------------------------------------------------------
    #  Equilibrium state
    # ------------------------------------------------------------------------------

    def equilibrium_state(self, q, xyz, loads, structure):
        """
        Assembles an equilibrium state object.
        """
        connectivity = structure.connectivity

        vectors = self.edges_vectors(xyz, connectivity)
        residuals = self.nodes_residuals(q, loads, vectors, connectivity)
        lengths = self.edges_lengths(vectors)
        forces = self.edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                loads=loads,
                                force_densities=q)

    # ----------------------------------------------------------------------
    # Matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        The stiffness matrix of the structure.
        """
        # shorthand
        c_free = structure.connectivity_free

        return c_free.T @ (q[:, None] * c_free)

    @staticmethod
    def force_matrix(q, xyz_fixed, loads, structure):
        """
        The force residual matrix of the structure.
        """
        # shorthands
        c_free = structure.connectivity_free
        c_fixed = structure.connectivity_fixed
        free = structure.indices_free

        return loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))


# ==========================================================================
# Sparse equilibrium model
# ==========================================================================

class EquilibriumModelSparse(EquilibriumModel):
    """
    The equilibrium model. Sparse.
    """
    def nodes_free_positions(self, q, xyz_fixed, loads, structure):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        A = self.stiffness_matrix(q, structure)
        b = self.force_matrix(q, xyz_fixed, loads, structure)

        return sparse_solve(A, b)

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        Computes the LHS matrix in CSC format from a vector of force densities.
        """
        # short hands
        index_array = structure.index_array
        diag_indices = structure.diag_indices
        diags = structure.diags

        nondiags_data = -q[index_array.data - 1]
        args = (nondiags_data, index_array.indices, index_array.indptr)

        nondiags = CSC(args, shape=index_array.shape)

        # sum of force densities for each node
        diag_fd = diags.T @ q

        nondiags.data = nondiags.data.at[diag_indices].set(diag_fd)

        return nondiags
