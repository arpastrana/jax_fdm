from chex import assert_max_traces
from functools import partial

from equinox import is_array

from jax.debug import print as jax_print

import jax.numpy as jnp

from jax.experimental.sparse import CSC

from jax_fdm.equilibrium.states import EquilibriumState

from jax_fdm.equilibrium.sparse import sparse_solve as spsolve

from jax_fdm.equilibrium.solvers import fixed_point
from jax_fdm.equilibrium.solvers import least_squares
from jax_fdm.equilibrium.solvers import is_solver_fixedpoint
from jax_fdm.equilibrium.solvers import is_solver_leastsquares
from jax_fdm.equilibrium.solvers import SOLVERS

from jax_fdm.equilibrium.loads import nodes_load_from_faces
from jax_fdm.equilibrium.loads import nodes_load_from_edges


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    A FDM model to calculate equilibrium states with shape-dependent loads.

    Parameters
    ----------
    `tmax`: The maximum number of iterations to calculate an equilibrium state. If `tmax=1`, the model is equivalent to doing one linear FDM step, and the rest of the parameters of this model are ignored. The edge and face loads are discarded too. Defaults to `100`.
    `eta`: The convergence tolerance for calculating an equilibrium state. Defaults to `1e-6`.
    `is_load_local`: If set to `True`, the face and edge loads are applied in their local coordinate system at every iteration (follower loads). Defaults to `False`.
    `itersolve_fn`: The function that calculates an equilibrium state iteratively. If `None`, the model defaults to forward fixed-point iteration. Note that only the solver must be consistent with the choice of residual function. Defaults to `None`.
    `implicit_diff`: If set to `True`, it applies implicit differentiation to speed up backpropagation. Defaults to `True`.
    `verbose`: Whether to print out calculations' info to the terminal. Defaults to `False`.
    """
    def __init__(self,
                 tmax=100,
                 eta=1e-6,
                 is_load_local=False,
                 itersolve_fn="fixedpoint",
                 implicit_diff=True,
                 verbose=False):
        self.tmax = tmax
        self.eta = eta
        self.is_load_local = is_load_local
        self.linearsolve_fn = jnp.linalg.solve
        self.verbose = verbose

        solver = self.assemble_solver(itersolve_fn, tmax, eta, implicit_diff, verbose)
        self.itersolve_fn = solver

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
    def nodes_positions(xyz_free, xyz_fixed, structure):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        indices = structure.indices_freefixed

        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def nodes_free_positions(self, q, xyz_fixed, loads, structure):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        K = self.stiffness_matrix(q, structure)
        P = self.load_matrix(q, xyz_fixed, loads, structure)

        return self.linearsolve_fn(K, P)

    def nodes_equilibrium(self, q, xyz_fixed, loads_nodes, structure):
        """
        Calculate static equilibrium on the nodes of a structure.
        """
        return self.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)

    # --------------------------------------------------------------------
    #  Load it up!
    # --------------------------------------------------------------------

    def nodes_load(self, xyz, load_state, structure):
        """
        Calculate the loads applied to the nodes of the structure.
        """
        nodes_load, edges_load, faces_load = load_state

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
        xyz_free = self.equilibrium(params, structure)

        if self.tmax > 1:
            xyz_free = self.equilibrium_iterative(params, structure, xyz_free)

        # Exit like a champ
        q, xyz_fixed, loads_state = params
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)
        loads_nodes = self.nodes_load(xyz, loads_state, structure)

        return self.equilibrium_state(q, xyz, loads_nodes, structure)

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(self, params, structure):
        """
        Calculate static equilibrium on a structure.
        """
        q, xyz_fixed, load_state = params
        loads_nodes = load_state.nodes

        return self.nodes_equilibrium(q, xyz_fixed, loads_nodes, structure)

    def equilibrium_iterative(self, params, structure, xyz_free_init):
        """
        Calculate static equilibrium on a structure iteratively.

        Notes
        -----
        This function only supports reverse mode auto-differentiation.
        To support forward-mode, we should define a custom jvp using implicit differentiation.
        """
        # Flatten XYZ free for compatibility with the loss function
        xyz_free_init = xyz_free_init.ravel()

        xyz_free_star = self.itersolve_fn(
            x_init=xyz_free_init,
            theta=params,
            structure=structure)

        return jnp.reshape(xyz_free_star, (-1, 3))

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
    # Stiffness matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        The stiffness matrix of the structure.
        """
        c_free = structure.connectivity_free

        return c_free.T @ (q[:, None] * c_free)

    # ----------------------------------------------------------------------
    # Load matrices
    # ----------------------------------------------------------------------

    def load_matrix(self, q, xyz_fixed, load_nodes, structure):
        """
        The load matrix of the structure.
        """
        R_fixed = self.residual_fixed_matrix(q, xyz_fixed, structure)

        return load_nodes[structure.indices_free, :] - R_fixed

    def load_xyz_matrix(self, params, xyz_free, structure):
        """
        Calculate loads matrix of the free nodes of the system for shape-dependent loads.
        """
        # Unpack parameters
        q, xyz_fixed, load_state = params

        # Concatenate free and fixed xyz coordinates
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

        # Calculate shape-dependent loads with full xyz
        loads_nodes = self.nodes_load(xyz, load_state, structure)

        # Assemble load matrix
        return self.load_matrix(q, xyz_fixed, loads_nodes, structure)

    def load_xyz_matrix_from_r_fixed(self, params, xyz_free, structure):
        """
        Calculate loads matrix of the free nodes of the system for shape-dependent loads.
        """
        K, R_fixed, xyz_fixed, load_state = params

        # Concatenate free and fixed xyz coordinates
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

        # Calculate shape-dependent loads with full xyz
        loads_nodes = self.nodes_load(xyz, load_state, structure)

        # Assemble load matrix
        P = loads_nodes[structure.indices_free, :] - R_fixed

        return P

    # ----------------------------------------------------------------------
    # Load matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def residual_fixed_matrix(q, xyz_fixed, structure):
        """
        The load matrix with the contribution of the fixed nodes's residuals.
        """
        c_free = structure.connectivity_free
        c_fixed = structure.connectivity_fixed

        return c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    # ----------------------------------------------------------------------
    # Helper functions for iterative equilibrium
    # ----------------------------------------------------------------------

    def assemble_solver(self, solver_name, tmax, eta, implicit_diff, verbose):
        """
        """
        solver_config = {"tmax": tmax,
                         "eta": eta,
                         "implicit_diff": implicit_diff,
                         "verbose": verbose}

        solver_fn = SOLVERS.get(solver_name)
        if solver_fn is None:
            raise ValueError(f"Unsupported solver name: {solver_name}!")

        f = self.pick_equilibrium_fn(solver_fn)
        solver = solver_fn(f, solver_config)

        if implicit_diff:
            solver_implicit = self.pick_solver_implicit_fn(solver_fn)
            return partial(solver_implicit, solver=solver)

        return solver

    def pick_solver_implicit_fn(self, solver):
        """
        Pick the implicit differentiation wrapper that is compatible with the input solver.
        """
        if is_solver_fixedpoint(solver):
            return fixed_point
        if is_solver_leastsquares(solver):
            return least_squares

        raise ValueError(f"Solver {solver} is not supported!")

    def pick_equilibrium_fn(self, solver):
        """
        Pick the equilibrium function that is compatible with the input solver.
        """
        if is_solver_fixedpoint(solver):
            return self.xyz_free
        if is_solver_leastsquares(solver):
            return self.residuals_free

        raise ValueError(f"Solver {solver} is not supported!")

    def calculate_kp_matrices(self, xyz_free, params, structure):
        """
        """
        # Unpack parameters
        q, xyz_fixed, load_state = params

        # Calculate stiffness matrix
        K = self.stiffness_matrix(q, structure)

        # Calculate load matrix
        P = self.load_xyz_matrix(params, xyz_free, structure)

        return K, P

    def xyz_free(self, xyz_free, params, structure):
        """
        The residual function of the equilibrium problem.
        """
        # Reformat xyz free
        xyz_free = jnp.reshape(xyz_free, (-1, 3))

        # Calculate matrices
        K, P = self.calculate_kp_matrices(xyz_free, params, structure)

        # Calculate xyz free
        xyz_free = self.linearsolve_fn(K, P)

        return xyz_free.ravel()

    def residuals_free(self, xyz_free, params, structure):
        """
        The residual function of the equilibrium problem.
        """
        # Reformat xyz free
        xyz_free = jnp.reshape(xyz_free, (-1, 3))

        # Calculate matrices
        K, P = self.calculate_kp_matrices(xyz_free, params, structure)

        # Residual function
        residual = K @ xyz_free - P

        return residual.ravel()


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
