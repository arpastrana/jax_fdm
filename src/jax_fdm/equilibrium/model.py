from functools import partial

import numpy as np

import jax.numpy as jnp

from jax_fdm.equilibrium.state import EquilibriumState
from jax_fdm.equilibrium.structure import EquilibriumStructure

from jax import jit

import equinox as eqx

# ==========================================================================
# Equilibrium model
# ==========================================================================


class EquilibriumModel:
    """
    The calculator.
    """
    # structure: None
    # loads: jnp.ndarray
    # xyz_fixed: jnp.ndarray

    def __init__(self, network):
        self.structure = EquilibriumStructure(network)
        self.loads = np.asarray(list(network.nodes_loads()), dtype=np.float64)
        self.xyz_fixed = np.asarray([network.node_coordinates(node) for node in network.nodes_fixed()], dtype=np.float64)

    @partial(jit, static_argnums=0)
    def _edges_lengths(self, xyz):
        connectivity = self.structure.connectivity
        return jnp.linalg.norm(connectivity @ xyz, axis=1)

    @partial(jit, static_argnums=0)
    def _edges_forces(self, q, lengths):
        return q * lengths

    @partial(jit, static_argnums=0)
    def _nodes_residuals(self, q, xyz):
        connectivity = self.structure.connectivity
        return self.loads - jnp.transpose(connectivity) @ jnp.diag(q) @ connectivity @ xyz

    @partial(jit, static_argnums=0)
    def _nodes_free_positions(self, q):
        # convenience shorthands
        connectivity = self.structure.connectivity
        free = self.structure.free_nodes
        fixed = self.structure.fixed_nodes
        loads = self.loads
        xyz_fixed = self.xyz_fixed

        # connectivity
        c_matrix = connectivity
        c_fixed = c_matrix[:, fixed]
        c_free = c_matrix[:, free]
        c_free_t = np.transpose(c_free)

        # Mutable stuff
        q_matrix = jnp.diag(q)

        # solve equilibrium after solving a linear system of equations
        A = c_free_t @ q_matrix @ c_free
        b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz_fixed

        return jnp.linalg.solve(A, b)

    @partial(jit, static_argnums=0)
    def _nodes_positions(self, xyz_free):
        # NOTE: free fixed indices sorted by enumeration
        xyz_fixed = self.xyz_fixed
        indices = self.structure.freefixed_nodes
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    # @partial(jit, static_argnums=0)
    def __call__(self, q):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz_free = self._nodes_free_positions(q)
        xyz = self._nodes_positions(xyz_free)
        residuals = self._nodes_residuals(q, xyz)
        lengths = self._edges_lengths(xyz)
        forces = self._edges_forces(q, lengths)

        return forces

        # return EquilibriumState(xyz=xyz,
        #                         residuals=residuals,
        #                         lengths=lengths,
        #                         forces=forces,
        #                         force_densities=q)
