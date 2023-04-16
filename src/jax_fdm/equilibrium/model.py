from functools import partial

from jax import jit
from jax import vmap

import jax.numpy as jnp

from equinox.internal import while_loop

from jax_fdm.datastructures import FDNetwork
from jax_fdm.datastructures import FDMesh

from jax_fdm.equilibrium.state import EquilibriumState
from jax_fdm.equilibrium.structure import EquilibriumStructure
from jax_fdm.equilibrium.structure import EquilibriumStructureMesh
from jax_fdm.equilibrium.loads import LoadCalculator


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    The equilibrium solver.
    """
    def __init__(self, datastruct):
        self.structure = datastruct
        self.load_calc = LoadCalculator(self.structure)

    # ------------------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------------------

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, datastruct):
        if isinstance(datastruct, FDNetwork):
            structure = EquilibriumStructure(datastruct)
        elif isinstance(datastruct, FDMesh):
            structure = EquilibriumStructureMesh(datastruct)
        else:
            raise ValueError(f"{datastruct} is not a valid datastructure!")

        self._structure = structure

    # ------------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------------

    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from a force density network.
        """
        return cls(network)

    @classmethod
    def from_mesh(cls, mesh):
        """
        Create an equilibrium model from a force density mesh.
        """
        return cls(mesh)

    # ------------------------------------------------------------------------------
    #  Edges
    # ------------------------------------------------------------------------------

    def edges_vector(self, xyz):
        """
        Calculate the unnormalized edge directions.
        """
        return self.structure.connectivity @ xyz

    def edges_length(self, vectors):
        """
        Compute the length of the edges.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    def edges_force(self, q, lengths):
        """
        Calculate the force in the edges.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    # ------------------------------------------------------------------------------
    #  Nodes
    # ------------------------------------------------------------------------------

    def nodes_residual(self, q, loads, vectors):
        """
        Compute the force at the anchor supports of the structure.
        """
        connectivity = self.structure.connectivity
        return loads - connectivity.T @ jnp.diag(q) @ vectors

    def nodes_free_position(self, q, xyz_fixed, loads):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        # shorthand
        free = self.structure.free_nodes
        c_fixed = self.structure.connectivity_fixed
        c_free = self.structure.connectivity_free

        A = c_free.T @ vmap(jnp.dot)(q, c_free)
        b = loads[free, :] - c_free.T @ vmap(jnp.dot)(q, c_fixed @ xyz_fixed)

        return jnp.linalg.solve(A, b)

    def nodes_position(self, xyz_free, xyz_fixed):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        # NOTE: free fixed indices sorted by enumeration
        indices = self.structure.freefixed_nodes
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def nodes_equilibrium(self, q, xyz_fixed, loads):
        """
        Calculate static equilibrium on the nodes of a structure.
        """
        xyz_free = self.nodes_free_position(q, xyz_fixed, loads)

        return self.nodes_position(xyz_free, xyz_fixed)

    # ------------------------------------------------------------------------------
    #  Loads
    # ------------------------------------------------------------------------------

    def nodes_load(self, xyz, loads):
        """
        Calculate the current loads applied to the nodes of the structure.
        """
        return self.load_calc(xyz, loads)

    # ------------------------------------------------------------------------------
    #  Call me, maybe
    # ------------------------------------------------------------------------------

    @partial(jit, static_argnums=(0, 4, 5))
    def __call__(self, q, xyz_fixed, loads, tmax=100, eta=1e-6):
        """
        Compute an equilibrium state using the force density method (FDM).
        """
        xyz = self.equilibrium(q, xyz_fixed, loads)

        if tmax > 0:
            xyz, loads = self.equilibrium_iterative(q, xyz_fixed, loads, tmax, eta)

        return self.equilibrium_state(q, xyz, loads)

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(self, q, xyz_fixed, loads):
        """
        Calculate static equilibrium on a structure.
        """
        return self.nodes_equilibrium(q, xyz_fixed, loads)

    def equilibrium_iterative(self, q, xyz_fixed, loads, tmax, eta):
        """
        Calculate static equilibrium on a structure iteratively.
        """

        def distance(xyz, xyz_last):
            return jnp.sum(jnp.linalg.norm(xyz_last[:-1] - xyz[:-1], axis=1))

        def cond_fn(val):
            xyz_last, xyz, _ = val
            # calculate residual distance
            residual = distance(xyz, xyz_last)
            # if residual distance larger than threshold, continue iterating
            return residual > eta

        def body_fn(val):
            _, xyz, _ = val
            xyz_last = xyz
            loads_upd = self.nodes_load(xyz, loads)
            xyz = self.nodes_equilibrium(q, xyz_fixed, loads_upd)
            return xyz_last, xyz, loads_upd

        xyz = self.nodes_equilibrium(q, xyz_fixed, loads)
        init_val = xyz * 1e9, xyz, loads
        _, xyz, loads = while_loop(cond_fn, body_fn, init_val, max_steps=tmax, kind="checkpointed")

        return xyz, loads

    # ------------------------------------------------------------------------------
    #  Equilibrium state
    # ------------------------------------------------------------------------------

    def equilibrium_state(self, q, xyz, loads):
        """
        Assembles an equilibrium state object.
        """
        vectors = self.edges_vector(xyz)
        residuals = self.nodes_residual(q, loads, vectors)
        lengths = self.edges_length(vectors)
        forces = self.edges_force(q, lengths)

        return EquilibriumState(xyz=xyz,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                loads=loads,
                                force_densities=q)
