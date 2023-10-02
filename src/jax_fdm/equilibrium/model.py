import jax.numpy as jnp

from jax_fdm.equilibrium.state import EquilibriumState

from jax_fdm.equilibrium.sparse import sparse_solve


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    The equilibrium solver.
    """
    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from a force density network.

        This method exists primarily for compatibility with older versions of the library.
        """
        return cls()

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

    @staticmethod
    def nodes_free_positions(q, xyz_fixed, loads, structure):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        # shorthand
        free = structure.nodes_indices_free
        c_fixed = structure.connectivity_fixed
        c_free = structure.connectivity_free

        A = c_free.T @ (q[:, None] * c_free)
        b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

        return jnp.linalg.solve(A, b)

    def __call__(self, params, structure):
        """
        Compute an equilibrium state using the force density method.
        """
        q, xyz_fixed, loads = params
        connectivity = structure.connectivity
        indices = structure.nodes_indices_freefixed

        xyz_free = self.nodes_free_positions(q, xyz_fixed, loads, structure)
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
    @staticmethod
    def nodes_free_positions(q, xyz_fixed, loads, structure):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        return sparse_solve(q, xyz_fixed, loads, structure)
