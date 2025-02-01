from equinox import is_array

from jax.debug import print as jax_print

import jax.numpy as jnp

from jax.experimental.sparse import CSC

from jax_fdm.equilibrium.states import EquilibriumState

from jax_fdm.equilibrium.sparse import sparse_solve as spsolve

from jax_fdm.equilibrium.solvers import solver_forward
from jax_fdm.equilibrium.solvers import fixed_point
from jax_fdm.equilibrium.solvers import least_squares
from jax_fdm.equilibrium.solvers import is_solver_fixedpoint
from jax_fdm.equilibrium.solvers import is_solver_leastsquares

from jax_fdm.equilibrium.loads import nodes_load_from_faces
from jax_fdm.equilibrium.loads import nodes_load_from_edges

from jax_fdm.equilibrium.states import LoadState


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
    `ignore_nodes_load`: Whether to only apply edge and face loads during the iterative equilibrium calculations. Defaults to `True`.
    `verbose`: Whether to print out calculations' info to the terminal. Defaults to `False`.
    """
    def __init__(self,
                 tmax=100,
                 eta=1e-6,
                 is_load_local=False,
                 itersolve_fn=None,
                 implicit_diff=True,
                 ignore_nodes_load=True,
                 verbose=False):
        self.tmax = tmax
        self.eta = eta
        self.is_load_local = is_load_local
        self.linearsolve_fn = jnp.linalg.solve
        self.itersolve_fn = itersolve_fn or solver_forward
        self.eq_iterative_fn = self.select_equilibrium_iterative_fn(self.itersolve_fn)
        self.implicit_diff = implicit_diff
        self.ignore_nodes_load = ignore_nodes_load
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
        A = self.stiffness_matrix(q, structure)
        b = self.force_matrix(q, xyz_fixed, loads, structure)

        return self.linearsolve_fn(A, b)

    def nodes_equilibrium(self, q, xyz_fixed, loads_nodes, structure):
        """
        Calculate static equilibrium on the nodes of a structure.
        """
        return self.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)

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

        xyz_free = self.equilibrium(q, xyz_fixed, loads_nodes, structure)

        if tmax > 1:

            # Setting node loads to zero when tmax > 1 if specified
            if self.ignore_nodes_load:
                loads_nodes = jnp.zeros_like(loads_nodes)
                loads_state = LoadState(loads_nodes,
                                        loads_state.edges,
                                        loads_state.faces)

            xyz_free = self.eq_iterative_fn(
                q,
                xyz_fixed,
                loads_state,
                structure,
                xyz_free_init=xyz_free,
                tmax=tmax,
                eta=eta,
                solver=solver,
                implicit_diff=implicit_diff,
                verbose=verbose
            )

        # Exit like a champ
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)
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

    def select_equilibrium_iterative_fn(self, solver):
        """
        Pick the equilibrium function that is compatible with the input solver function.
        """
        if is_solver_fixedpoint(solver):
            return self.equilibrium_iterative_xyz

        if is_solver_leastsquares(solver):
            return self.equilibrium_iterative_residual

        raise ValueError(f"Solver {solver} is not supported!")

    def equilibrium_iterative_xyz(
            self,
            q,
            xyz_fixed,
            load_state,
            structure,
            xyz_free_init=None,
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
        def equilibrium_iterative_fn(params, xyz_free):
            """
            This closure function avoids re-computing A and f_fixed throughout iterations
            because these two matrices remain constant during the fixed point search.
            """
            # TODO: Extract closure into function shared with the other nodes equilibrium function?
            A, f_fixed, xyz_fixed, load_state = params

            xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)
            loads_nodes = self.nodes_load(xyz, load_state, structure)
            b = loads_nodes[structure.indices_free, :] - f_fixed

            return self.linearsolve_fn(A, b)

        # recompute xyz_init if not input
        if xyz_free_init is None:
            xyz_free_init = self.equilibrium(q, xyz_fixed, load_state.nodes, structure)

        A = self.stiffness_matrix(q, structure)
        f_fixed = self.force_fixed_matrix(q, xyz_fixed, structure)

        solver = solver or self.itersolve_fn
        solver_config = {"tmax": tmax,
                         "eta": eta,
                         "implicit_diff": implicit_diff,
                         "verbose": verbose}

        solver_kwargs = {"solver_config": solver_config,
                         "f": equilibrium_iterative_fn,
                         "a": (A, f_fixed, xyz_fixed, load_state),
                         "x_init": xyz_free_init}

        if implicit_diff:
            return fixed_point(solver, **solver_kwargs)

        return solver(**solver_kwargs)

    def equilibrium_iterative_residual(self,
                                       q,
                                       xyz_fixed,
                                       load_state,
                                       structure,
                                       xyz_free_init=None,
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
        def loads_fn(params, xyz_free):
            """
            Calculate loads matrix of the free nodes of the system.
            """
            # Unpack parameters
            q, xyz_fixed, load_state = params

            # Concatenate free and fixed xyz coordinates
            xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

            # Calculate shape-dependent loads with full xyz
            loads_nodes = self.nodes_load(xyz, load_state, structure)

            return self.force_matrix(q, xyz_fixed, loads_nodes, structure)

        def residual_fn(params, xyz_free):
            """
            The residual function of the equilibrium problem.
            """
            # Unpack parameters
            q, xyz_fixed, load_state = params

            # Calculate stiffness matrix
            A = self.stiffness_matrix(q, structure)

            # Calculate load matrix
            xyz_free = jnp.reshape(xyz_free, (-1, 3))
            b = loads_fn(params, xyz_free)

            # Residual function
            residual = A @ xyz_free - b

            return residual.ravel()

        # Recompute xyz_free_init if not input
        if xyz_free_init is None:
            load_nodes = load_state.nodes
            xyz_free_init = self.nodes_free_positions(q, xyz_fixed, load_nodes, structure)

        # Flatten XYZ free for compatibility with the loss function
        xyz_free_init = xyz_free_init.ravel()

        # Parameterss
        params = (q, xyz_fixed, load_state)

        # Solver
        solver_config = {"tmax": tmax,
                         "eta": eta,
                         "implicit_diff": implicit_diff,
                         "verbose": verbose}

        solver_kwargs = {"solver_config": solver_config,
                         "f": residual_fn,
                         "a": params,
                         "x_init": xyz_free_init}

        solver = solver or self.itersolve_fn
        # if implicit_diff:
        #    xyz_free_star = fixed_point(solver, **solver_kwargs)
        # else:
        print("Solving equilibrium problem...")
        xyz_free_star = solver(**solver_kwargs)

        residual = residual_fn(params, xyz_free_star)
        residual = jnp.linalg.norm(jnp.reshape(residual, (-1, 3)), axis=1)
        print(f"Mean solution residual: {jnp.mean(residual).item():.3f}")

        xyz_free_star = jnp.reshape(xyz_free_star, (-1, 3))

        return xyz_free_star

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

    def force_matrix(self, q, xyz_fixed, load_nodes, structure):
        """
        The force residual matrix of the structure.
        """
        free = structure.indices_free

        return load_nodes[free, :] - self.force_fixed_matrix(q, xyz_fixed, structure)

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
