from equinox import is_array

import jax.numpy as jnp

from jax.experimental.sparse import CSC

from jax_fdm.equilibrium.states import EquilibriumState

from jax_fdm.equilibrium.sparse import sparse_solve as spsolve

from jax_fdm.equilibrium.solvers import fixed_point
from jax_fdm.equilibrium.solvers import solver_forward

from jax_fdm.equilibrium.loads import nodes_load_from_faces
from jax_fdm.equilibrium.loads import nodes_load_from_edges

from jax_fdm.equilibrium.states import LoadState


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    The equilibrium model.
    """
    def __init__(self,
                 tmax=100,
                 eta=1e-6,
                 is_load_local=False,
                 itersolve_fn=None,
                 implicit_diff=True,
                 load_nodes_iter=False,
                 verbose=False):

        self.tmax = tmax
        self.eta = eta
        self.is_load_local = is_load_local
        self.linearsolve_fn = jnp.linalg.solve
        self.itersolve_fn = itersolve_fn or solver_forward
        self.implicit_diff = implicit_diff
        self.load_nodes_iter = load_nodes_iter
        self.verbose = verbose

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

        return self.linearsolve_fn(A, b)

    def nodes_equilibrium(self, q, xyz_fixed, loads_nodes, structure):
        """
        Calculate static equilibrium on the nodes of a structure.
        """
        indices = structure.indices_freefixed
        xyz_free = self.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)

        return self.nodes_positions(xyz_free, xyz_fixed, indices)

    # --------------------------------------------------------------------
    #  Load it up!
    # --------------------------------------------------------------------

    def nodes_load(self, xyz, loads, structure):
        """
        Calculate the loads applied to the nodes of the structure.
        """
        nodes_load, edges_load, faces_load = loads

        if is_array(edges_load):
            if edges_load.size > 1:
                edges_load_ = self.edges_load(xyz, edges_load, structure, self.is_load_local)
                nodes_load = nodes_load + edges_load_

        if is_array(faces_load):
            if faces_load.size > 1:
                faces_load_ = self.faces_load(xyz, faces_load, structure, self.is_load_local)
                nodes_load = nodes_load + faces_load_

        return nodes_load

    @staticmethod
    def faces_load(xyz, faces_load, structure, is_local=False):
        """
        Calculate the tributary face loads aplied to the nodes of a structure.
        """
        return nodes_load_from_faces(xyz, faces_load, structure, is_local)

    @staticmethod
    def edges_load(xyz, edges_load, structure, is_local=False):
        """
        Calculate the tributary edge loads aplied to the nodes of a structure.
        """
        return nodes_load_from_edges(xyz, edges_load, structure, is_local)

    # ------------------------------------------------------------------------------
    #  Call me, maybe
    # ------------------------------------------------------------------------------

    def __call__(self, params, structure):
        """
        Compute an equilibrium state using the force density method (FDM).
        """
        q, xyz_fixed, loads_state = params
        loads_nodes = loads_state.nodes

        tmax = self.tmax
        eta = self.eta
        solver = self.itersolve_fn
        implicit_diff = self.implicit_diff
        verbose = self.verbose

        xyz = self.equilibrium(q, xyz_fixed, loads_nodes, structure)

        if tmax > 1:
            # Setting node loads to zero when tmax > 1 if specified
            if not self.load_nodes_iter:
                loads_nodes = jnp.zeros_like(loads_nodes)
                loads_state = LoadState(loads_nodes,
                                        loads_state.edges,
                                        loads_state.faces)

            xyz = self.equilibrium_iterative(q,
                                             xyz_fixed,
                                             loads_state,
                                             structure,
                                             xyz_init=xyz,
                                             tmax=tmax,
                                             eta=eta,
                                             solver=solver,
                                             implicit_diff=implicit_diff,
                                             verbose=verbose)

        loads_nodes = self.nodes_load(xyz, loads_state, structure)

        return self.equilibrium_state(q, xyz, loads_nodes, structure)

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(self, q, xyz_fixed, loads_nodes, structure):
        """
        Calculate static equilibrium on a structure.
        """
        return self.nodes_equilibrium(q, xyz_fixed, loads_nodes, structure)

    def equilibrium_iterative(self,
                              q,
                              xyz_fixed,
                              load_state,
                              structure,
                              xyz_init=None,
                              tmax=100,
                              eta=1e-6,
                              solver=None,
                              implicit_diff=True,
                              verbose=False):
        """
        Calculate static equilibrium on a structure iteratively.

        Notes
        -----
        This function only supports reverse mode auto-differentiation.
        To support forward-mode, we should define a custom jvp using implicit differentiation.
        """
        def equilibrium_iterative_fn(params, xyz_init):
            """
            This closure function avoids re-computing A and f_fixed throughout iterations
            because these two matrices remain constant during the fixed point search.

            TODO: Extract closure into function shared with the other nodes equilibrium function?
            """
            A, f_fixed, xyz_fixed, load_state = params

            free = structure.indices_free
            freefixed = structure.indices_freefixed

            loads_nodes = self.nodes_load(xyz_init, load_state, structure)
            b = loads_nodes[free, :] - f_fixed
            xyz_free = self.linearsolve_fn(A, b)
            xyz_ = self.nodes_positions(xyz_free, xyz_fixed, freefixed)

            return xyz_

        # recompute xyz_init if not input
        if xyz_init is None:
            xyz_init = self.equilibrium(q, xyz_fixed, load_state.nodes, structure)

        A = self.stiffness_matrix(q, structure)
        f_fixed = self.force_fixed_matrix(q, xyz_fixed, structure)

        solver = solver or self.itersolve_fn
        solver_config = {"tmax": tmax,
                         "eta": eta,
                         "verbose": verbose,
                         # For jaxopt compatibility (as it does not support sparse matrices yet)
                         "implicit": False if self.linearsolve_fn is spsolve else True}

        solver_kwargs = {"solver_config": solver_config,
                         "f": equilibrium_iterative_fn,
                         "a": (A, f_fixed, xyz_fixed, load_state),
                         "x_init": xyz_init}

        if implicit_diff:
            xyz_new = fixed_point(solver, **solver_kwargs)

        xyz_new = solver(**solver_kwargs)

        return xyz_new

    # ----------------------------------------------------------------------
    # Equilibrium state
    # ----------------------------------------------------------------------

    def equilibrium_state(self, q, xyz, loads_nodes, structure):
        """
        Assembles an equilibrium state object.
        """
        connectivity = structure.connectivity

        vectors = self.edges_vectors(xyz, connectivity)
        lengths = self.edges_lengths(vectors)
        residuals = self.nodes_residuals(q, loads_nodes, vectors, connectivity)
        forces = self.edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                loads=loads_nodes,
                                vectors=vectors)

    # ----------------------------------------------------------------------
    # Matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        The stiffness matrix of the structure.
        """
        c_free = structure.connectivity_free

        return c_free.T @ (q[:, None] * c_free)

    def force_matrix(self, q, xyz_fixed, loads, structure):
        """
        The force residual matrix of the structure.
        """
        free = structure.indices_free

        return loads[free, :] - self.force_fixed_matrix(q, xyz_fixed, structure)

    @staticmethod
    def force_fixed_matrix(q, xyz_fixed, structure):
        """
        The force matrix block of the residual forces at the fixed nodes.
        """
        c_free = structure.connectivity_free
        c_fixed = structure.connectivity_fixed

        return c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))


# ==========================================================================
# Sparse equilibrium model
# ==========================================================================

class EquilibriumModelSparse(EquilibriumModel):
    """
    The equilibrium model. Sparse.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearsolve_fn = spsolve

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        Computes the LHS matrix in CSC format from a vector of force densities.
        """
        # shorthands
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
