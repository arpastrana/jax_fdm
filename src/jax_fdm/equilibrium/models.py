from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.debug import print as jax_print
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import CSC
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium.loads import nodes_load_from_edges
from jax_fdm.equilibrium.loads import nodes_load_from_faces
from jax_fdm.equilibrium.solvers import is_solver_fixedpoint
from jax_fdm.equilibrium.solvers import is_solver_leastsquares
from jax_fdm.equilibrium.solvers import is_solver_root_finding
from jax_fdm.equilibrium.solvers import solver_fixedpoint_implicit
from jax_fdm.equilibrium.solvers import solver_forward
from jax_fdm.equilibrium.solvers import solver_nonlinear_implicit
from jax_fdm.equilibrium.sparse import sparse_solve as spsolve
from jax_fdm.equilibrium.states import EquilibriumParametersState
from jax_fdm.equilibrium.states import EquilibriumState
from jax_fdm.equilibrium.states import LoadState
from jax_fdm.equilibrium.structures import EquilibriumMeshStructure
from jax_fdm.equilibrium.structures import EquilibriumStructure
from jax_fdm.equilibrium.structures import EquilibriumStructureSparse

# ==========================================================================
# Type aliases
# ==========================================================================

# The stiffness matrix K, dense on the base model and CSC on the sparse model.
StiffnessMatrix = (
    Float[Array, "nodes_free nodes_free"] | Float[CSC, "nodes_free nodes_free"]
)

# ==========================================================================
# Equilibrium model
# ==========================================================================


class EquilibriumModel:
    """
    A FDM model to calculate equilibrium states with shape-dependent loads.

    Parameters
    ----------
    tmax :
        The maximum number of iterations to calculate an equilibrium state.
        If ``tmax=1``, the model does one linear FDM step and the remaining
        parameters are ignored; the edge and face loads are discarded too.
    eta :
        The convergence tolerance for calculating an equilibrium state.
    is_load_local :
        If True, the face and edge loads are applied in their local
        coordinate system at every iteration (follower loads).
    itersolve_fn :
        The function that calculates an equilibrium state iteratively.
        If None, defaults to forward fixed-point iteration. The solver must be
        consistent with the choice of residual function.
    iterload_fn :
        A load callback invoked once before iterative equilibrium starts.
        If None, the load state is left untouched.
    implicit_diff :
        If True, apply implicit differentiation to speed up backpropagation.
    verbose :
        Whether to print calculation info to the terminal.
    """

    def __init__(
        self,
        tmax: int = 100,
        eta: float = 1e-6,
        is_load_local: bool = False,
        itersolve_fn: Callable | None = None,
        iterload_fn: Callable | None = None,
        implicit_diff: bool = True,
        verbose: bool = False,
    ):
        self.tmax = tmax
        self.eta = eta
        self.is_load_local = is_load_local
        self.linearsolve_fn = jnp.linalg.solve
        self.itersolve_fn = itersolve_fn or solver_forward
        self.iterload_fn = iterload_fn
        self.eq_iterative_fn = self.select_equilibrium_iterative_fn(self.itersolve_fn)
        self.implicit_diff = implicit_diff
        self.verbose = verbose

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    @staticmethod
    def edges_vectors(
        xyz: Float[Array, "nodes 3"],
        connectivity: Float[Array, "edges nodes"] | Float[BCOO, "edges nodes"],
    ) -> Float[Array, "edges 3"]:
        """
        Calculate the unnormalized edge vectors as nodal coordinate differences.

        Parameters
        ----------
        xyz :
            The coordinates of the nodes.
        connectivity :
            The signed edge-node incidence matrix of the structure.

        Returns
        -------
        vectors :
            The edge vectors pointing from tail to head node.
        """
        return connectivity @ xyz

    @staticmethod
    def edges_lengths(vectors: Float[Array, "edges 3"]) -> Float[Array, "edges 1"]:
        """
        Compute the length of the edges.

        Parameters
        ----------
        vectors :
            The edge vectors.

        Returns
        -------
        lengths :
            The Euclidean length of each edge.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    @staticmethod
    def edges_forces(
        q: Float[Array, "edges"],
        lengths: Float[Array, "edges 1"],
    ) -> Float[Array, "edges 1"]:
        """
        Calculate the axial force in the edges.

        Parameters
        ----------
        q :
            The force densities of the edges.
        lengths :
            The lengths of the edges.

        Returns
        -------
        forces :
            The axial force in each edge, the product of force density and length.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    # ----------------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------------

    @staticmethod
    def nodes_residuals(
        q: Float[Array, "edges"],
        loads: Float[Array, "nodes 3"],
        vectors: Float[Array, "edges 3"],
        connectivity: Float[Array, "edges nodes"] | Float[BCOO, "edges nodes"],
    ) -> Float[Array, "nodes 3"]:
        """
        Compute the residual forces on the nodes of the structure.

        Parameters
        ----------
        q :
            The force densities of the edges.
        loads :
            The loads applied to the nodes.
        vectors :
            The edge vectors.
        connectivity :
            The signed edge-node incidence matrix of the structure.

        Returns
        -------
        residuals :
            The residual force at each node, zero at nodes in equilibrium.
        """
        # upstream types the sparse transpose as optional; it is None only for
        # arrays of dimension > 2, unreachable for a rank-2 incidence matrix
        return loads - connectivity.T @ (q[:, None] * vectors)  # pyright: ignore[reportOptionalOperand]

    @staticmethod
    def nodes_positions(
        xyz_free: Float[Array, "nodes_free 3"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes 3"]:
        """
        Concatenate the free and fixed node positions back into node order.

        Parameters
        ----------
        xyz_free :
            The coordinates of the free nodes.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        structure :
            The structure whose free/fixed index map reorders the nodes.

        Returns
        -------
        xyz :
            The coordinates of all nodes, restored to the structure's node order.
        """
        indices = structure.indices_freefixed

        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def nodes_free_positions(
        self,
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        loads: Float[Array, "nodes 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Calculate the coordinates of the free nodes by solving the FDM system.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        loads :
            The loads applied to the nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        xyz_free :
            The coordinates of the free nodes at equilibrium.

        Notes
        -----
        Solves ``K @ xyz_free = P``, the linear FDM equilibrium system, with the
        stiffness matrix ``K`` and the load matrix ``P``.
        """
        K = self.stiffness_matrix(q, structure)
        P = self.load_matrix(q, xyz_fixed, loads, structure)

        return self.linearsolve_fn(K, P)

    def nodes_equilibrium(
        self,
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        loads_nodes: Float[Array, "nodes 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Calculate static equilibrium on the nodes of a structure.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        loads_nodes :
            The loads applied to the nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        xyz_free :
            The coordinates of the free nodes at equilibrium.
        """
        return self.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)

    # --------------------------------------------------------------------
    #  Load it up!
    # --------------------------------------------------------------------

    def nodes_load(
        self,
        xyz: Float[Array, "nodes 3"],
        load_state: LoadState,
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes 3"]:
        """
        Calculate the total load applied to the nodes of the structure.

        Parameters
        ----------
        xyz :
            The coordinates of all nodes.
        load_state :
            The nodal, edge, and face loads to aggregate onto the nodes.
        structure :
            The structure that provides the connectivity used to distribute
            edge and face loads to the nodes.

        Returns
        -------
        loads :
            The load at each node, including tributary edge and face loads.

        Notes
        -----
        Edge and face loads are added only when present (non-scalar). A non-scalar
        face load occurs only for meshes, so ``structure`` is a mesh structure in
        that branch.
        """
        nodes_load, edges_load, faces_load = load_state

        if isinstance(edges_load, jax.Array):
            if edges_load.size > 1:
                edges_load_ = self.edges_load(
                    xyz,
                    edges_load,
                    structure,
                    self.is_load_local,
                )
                nodes_load = nodes_load + edges_load_

        if isinstance(faces_load, jax.Array):
            if faces_load.size > 1:
                # A non-scalar faces_load only occurs for meshes
                # (LoadState.from_datastructure sets faces=0.0 for networks), so
                # structure is always an EquilibriumMeshStructure in this branch.
                faces_load_ = self.faces_load(
                    xyz,
                    faces_load,
                    structure,  # pyright: ignore[reportArgumentType]
                    self.is_load_local,
                )
                nodes_load = nodes_load + faces_load_

        return nodes_load

    @staticmethod
    def faces_load(
        xyz: Float[Array, "nodes 3"],
        faces_load: Float[Array, "faces 3"],
        structure: EquilibriumMeshStructure,
        is_local: bool = False,
    ) -> Float[Array, "nodes 3"]:
        """
        Distribute face area loads to the nodes of a structure.

        Parameters
        ----------
        xyz :
            The coordinates of all nodes.
        faces_load :
            The area load on each face.
        structure :
            The mesh structure that maps faces to their nodes.
        is_local :
            If True, the face load is applied in the face's local coordinate
            system (a follower load that tracks the deformed geometry).

        Returns
        -------
        loads :
            The tributary face load carried by each node.
        """
        return nodes_load_from_faces(xyz, faces_load, structure, is_local)

    @staticmethod
    def edges_load(
        xyz: Float[Array, "nodes 3"],
        edges_load: Float[Array, "edges 3"],
        structure: EquilibriumStructure,
        is_local: bool = False,
    ) -> Float[Array, "nodes 3"]:
        """
        Distribute edge line loads to the nodes of a structure.

        Parameters
        ----------
        xyz :
            The coordinates of all nodes.
        edges_load :
            The line load on each edge.
        structure :
            The structure that maps edges to their nodes.
        is_local :
            If True, the edge load is applied in the edge's local coordinate
            system (a follower load that tracks the deformed geometry).

        Returns
        -------
        loads :
            The tributary edge load carried by each node.
        """
        return nodes_load_from_edges(xyz, edges_load, structure, is_local)

    # ------------------------------------------------------------------------------
    #  Call me, maybe
    # ------------------------------------------------------------------------------

    def __call__(
        self,
        params: EquilibriumParametersState,
        structure: EquilibriumStructure,
    ) -> EquilibriumState:
        """
        Compute an equilibrium state using the force density method (FDM).

        Parameters
        ----------
        params :
            The force densities, fixed node coordinates, and load state.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        eq_state :
            The equilibrium state with node coordinates, residuals, edge lengths,
            edge forces, node loads, and edge vectors.

        Notes
        -----
        A single linear FDM step is taken first. When ``tmax > 1`` the solution is
        refined iteratively to account for shape-dependent loads, optionally with
        implicit differentiation for the backward pass.
        """
        q, xyz_fixed, load_state = params
        load_nodes = load_state.nodes

        tmax = self.tmax
        eta = self.eta
        solver = self.itersolve_fn
        iterload_fn = self.iterload_fn
        implicit_diff = self.implicit_diff
        verbose = self.verbose

        xyz_free = self.equilibrium(q, xyz_fixed, load_nodes, structure)

        if tmax > 1:
            if iterload_fn is not None:
                load_state = iterload_fn(load_state)
                params = EquilibriumParametersState(q, xyz_fixed, load_state)

            xyz_free = self.eq_iterative_fn(
                q,
                xyz_fixed,
                load_state,
                structure,
                xyz_free_init=xyz_free,
                tmax=tmax,
                eta=eta,
                solver=solver,
                implicit_diff=implicit_diff,
                verbose=verbose,
            )

        if self.verbose:
            residuals_free = self.residual_free_matrix(params, xyz_free, structure)
            jax_print(
                "Mean free residual vector: {}",
                jnp.mean(jnp.abs(residuals_free), axis=0),
            )

        # Exit like a champ
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)
        load_nodes = self.nodes_load(xyz, load_state, structure)

        return self.equilibrium_state(q, xyz, load_nodes, structure)

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(
        self,
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        loads_nodes: Float[Array, "nodes 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Calculate one linear force density step on a structure.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        loads_nodes :
            The loads applied to the nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        xyz_free :
            The coordinates of the free nodes at equilibrium.
        """
        return self.nodes_equilibrium(q, xyz_fixed, loads_nodes, structure)

    def equilibrium_iterative_xyz(
        self,
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        load_state: LoadState,
        structure: EquilibriumStructure,
        xyz_free_init: Float[Array, "nodes_free 3"] | None = None,
        tmax: int = 100,
        eta: float = 1e-6,
        solver: Callable | None = None,
        implicit_diff: bool = True,
        verbose: bool = False,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Calculate static equilibrium iteratively via fixed-point iteration on xyz.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        load_state :
            The nodal, edge, and face loads.
        structure :
            The structure that provides the connectivity matrices.
        xyz_free_init :
            The initial guess for the free node coordinates. If None, it is seeded
            with one linear FDM step.
        tmax :
            The maximum number of fixed-point iterations.
        eta :
            The convergence tolerance on the iterates.
        solver :
            The fixed-point solver. If None, the model's default solver is used.
        implicit_diff :
            If True, differentiate through the fixed point implicitly.
        verbose :
            Whether to print convergence info to the terminal.

        Returns
        -------
        xyz_free :
            The coordinates of the free nodes at the converged fixed point.

        Notes
        -----
        This function only supports reverse mode auto-differentiation.
        To support forward-mode, we should define a custom jvp using implicit
        differentiation.
        """

        def loads_fn(params, xyz_free):
            """
            A closure function over a structure to calculate the load matrix.
            This matrix is the RHS of the equilibrium linear system.
            """
            return self.load_xyz_matrix_from_r_fixed(params, xyz_free, structure)

        def equilibrium_iterative_fn(params, xyz_free):
            """
            Parameters
            ----------
            params :
                A tuple with parameters (K, R_fixed, xyz_fixed, load_state).
            xyz_free :
                The 3D coordinates of the free vertices.

            Returns
            -------
            xyz_free_updated :
                The updated 3D coordinates of the free vertices.
            """
            # Assemble load matrix
            P = loads_fn(params, xyz_free)

            # Fetch stiffness matrix
            K = params[0]

            return self.linearsolve_fn(K, P)

        if xyz_free_init is None:
            xyz_free_init = self.equilibrium(q, xyz_fixed, load_state.nodes, structure)

        K = self.stiffness_matrix(q, structure)
        R_fixed = self.residual_fixed_matrix(q, xyz_fixed, structure)

        solver_config = {
            "tmax": tmax,
            "eta": eta,
            "implicit_diff": implicit_diff,
            "verbose": verbose,
            "loads_fn": loads_fn,
        }

        solver_kwargs = {
            "solver_config": solver_config,
            "f": equilibrium_iterative_fn,
            "a": (K, R_fixed, xyz_fixed, load_state),
            "x_init": xyz_free_init,
        }

        solver = solver or self.itersolve_fn
        if implicit_diff:
            return solver_fixedpoint_implicit(solver, **solver_kwargs)

        return solver(**solver_kwargs)

    def equilibrium_iterative_residual(
        self,
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        load_state: LoadState,
        structure: EquilibriumStructure,
        xyz_free_init: Float[Array, "nodes_free 3"] | None = None,
        tmax: int = 100,
        eta: float = 1e-6,
        solver: Callable | None = None,
        implicit_diff: bool = True,
        verbose: bool = False,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Calculate static equilibrium iteratively by driving nodal residuals to zero.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        load_state :
            The nodal, edge, and face loads.
        structure :
            The structure that provides the connectivity matrices.
        xyz_free_init :
            The initial guess for the free node coordinates. If None, it is seeded
            with one linear FDM step.
        tmax :
            The maximum number of solver iterations.
        eta :
            The convergence tolerance on the residuals.
        solver :
            The least-squares or root-finding solver. If None, the model's default
            solver is used.
        implicit_diff :
            If True, differentiate through the solution implicitly.
        verbose :
            Whether to print convergence info to the terminal.

        Returns
        -------
        xyz_free :
            The coordinates of the free nodes where the residuals vanish.

        Notes
        -----
        This function only supports reverse mode auto-differentiation.
        To support forward-mode, we should define a custom jvp using implicit
        differentiation.
        """

        def residual_fn(params, xyz_free):
            """
            The residual function of the equilibrium problem.
            """
            xyz_free = jnp.reshape(xyz_free, (-1, 3))
            residuals = self.residual_free_matrix(params, xyz_free, structure)

            return residuals.ravel()

        # Recompute xyz_free_init if not input
        if xyz_free_init is None:
            xyz_free_init = self.equilibrium(q, xyz_fixed, load_state.nodes, structure)

        # Flatten XYZ free for compatibility with the loss function
        xyz_free_init = xyz_free_init.ravel()

        # Parameterss
        params = (q, xyz_fixed, load_state)

        # Solver
        solver_config = {
            "tmax": tmax,
            "eta": eta,
            "implicit_diff": False,
            "verbose": verbose,
        }

        solver_kwargs = {
            "solver_config": solver_config,
            "fn": residual_fn,
            "theta": params,
            "x_init": xyz_free_init,
        }

        solver = solver or self.itersolve_fn

        if implicit_diff:
            xyz_free_star = solver_nonlinear_implicit(solver, **solver_kwargs)
        else:
            xyz_free_star = solver(**solver_kwargs)

        # solver_nonlinear_implicit's return type is opaque to pyright (the
        # custom_vjp wrapper is defined in the out-of-scope solvers/nonlinear.py);
        # xyz_free_star is a jax.Array at runtime
        xyz_free_star = jnp.reshape(xyz_free_star, (-1, 3))  # pyright: ignore[reportArgumentType]

        return xyz_free_star

    def select_equilibrium_iterative_fn(self, solver: Callable) -> Callable:
        """
        Pick the iterative equilibrium function compatible with a solver.

        Parameters
        ----------
        solver :
            The iterative solver function to match.

        Returns
        -------
        equilibrium_fn :
            The fixed-point (xyz) function for fixed-point solvers, or the residual
            function for least-squares and root-finding solvers.

        Raises
        ------
        ValueError
            If the solver does not belong to a supported family.
        """
        if is_solver_fixedpoint(solver):
            return self.equilibrium_iterative_xyz

        if is_solver_leastsquares(solver) or is_solver_root_finding(solver):
            return self.equilibrium_iterative_residual

        raise ValueError(f"Solver {solver} is not supported!")

    # ----------------------------------------------------------------------
    # Equilibrium state
    # ----------------------------------------------------------------------

    def equilibrium_state(
        self,
        q: Float[Array, "edges"],
        xyz: Float[Array, "nodes 3"],
        loads_nodes: Float[Array, "nodes 3"],
        structure: EquilibriumStructure,
    ) -> EquilibriumState:
        """
        Assemble an equilibrium state from the equilibrated geometry.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz :
            The coordinates of all nodes.
        loads_nodes :
            The loads applied to the nodes.
        structure :
            The structure that provides the connectivity matrix.

        Returns
        -------
        eq_state :
            The equilibrium state bundling node coordinates, residuals, edge
            lengths, edge forces, node loads, and edge vectors.
        """
        connectivity = structure.connectivity

        vectors = self.edges_vectors(xyz, connectivity)
        lengths = self.edges_lengths(vectors)
        residuals = self.nodes_residuals(q, loads_nodes, vectors, connectivity)
        forces = self.edges_forces(q, lengths)

        return EquilibriumState(
            xyz=xyz,
            residuals=residuals,
            lengths=lengths,
            forces=forces,
            loads=loads_nodes,
            vectors=vectors,
        )

    # ----------------------------------------------------------------------
    # Stiffness matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def stiffness_matrix(
        q: Float[Array, "edges"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free nodes_free"]:
        """
        Assemble the force density stiffness matrix of the free nodes.

        Parameters
        ----------
        q :
            The force densities of the edges.
        structure :
            The structure that provides the free-node connectivity matrix.

        Returns
        -------
        stiffness :
            The stiffness matrix ``Cf.T @ diag(q) @ Cf`` of the free nodes.
        """
        c_free = structure.connectivity_free

        return c_free.T @ (q[:, None] * c_free)

    # ----------------------------------------------------------------------
    # Load matrices
    # ----------------------------------------------------------------------

    def load_matrix(
        self,
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        load_nodes: Float[Array, "nodes 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Assemble the right-hand side load matrix of the FDM system.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        load_nodes :
            The loads applied to the nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        load_matrix :
            The free-node loads minus the fixed nodes' residual contribution.
        """
        R_fixed = self.residual_fixed_matrix(q, xyz_fixed, structure)

        return load_nodes[structure.indices_free, :] - R_fixed

    def load_xyz_matrix(
        self,
        params: EquilibriumParametersState,
        xyz_free: Float[Array, "nodes_free 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Assemble the free-node load matrix for shape-dependent loads.

        Parameters
        ----------
        params :
            The force densities, fixed node coordinates, and load state.
        xyz_free :
            The current coordinates of the free nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        load_matrix :
            The right-hand side load matrix evaluated at the current geometry.

        Notes
        -----
        The full node coordinates are reassembled first so that edge and face
        loads can be recomputed against the current shape.
        """
        # Unpack parameters
        q, xyz_fixed, load_state = params

        # Concatenate free and fixed xyz coordinates
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

        # Calculate shape-dependent loads with full xyz
        loads_nodes = self.nodes_load(xyz, load_state, structure)

        # Assemble load matrix
        return self.load_matrix(q, xyz_fixed, loads_nodes, structure)

    def load_xyz_matrix_from_r_fixed(
        self,
        params: tuple[
            StiffnessMatrix,
            Float[Array, "nodes_free 3"],
            Float[Array, "nodes_fixed 3"],
            LoadState,
        ],
        xyz_free: Float[Array, "nodes_free 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Assemble the free-node load matrix reusing a precomputed fixed residual.

        Parameters
        ----------
        params :
            A tuple of the stiffness matrix, the precomputed fixed-node residual
            matrix, the fixed node coordinates, and the load state.
        xyz_free :
            The current coordinates of the free nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        load_matrix :
            The right-hand side load matrix evaluated at the current geometry.

        Notes
        -----
        This is the iteration hot path: the fixed-node residual is passed in rather
        than recomputed each step, unlike `load_xyz_matrix`.
        """
        _, R_fixed, xyz_fixed, load_state = params

        # Concatenate free and fixed xyz coordinates
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

        # Calculate shape-dependent loads with full xyz
        loads_nodes = self.nodes_load(xyz, load_state, structure)

        # Assemble load matrix
        P = loads_nodes[structure.indices_free, :] - R_fixed

        return P

    # ----------------------------------------------------------------------
    # Residual matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def residual_fixed_matrix(
        q: Float[Array, "edges"],
        xyz_fixed: Float[Array, "nodes_fixed 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Compute the fixed nodes' residual contribution to the free-node loads.

        Parameters
        ----------
        q :
            The force densities of the edges.
        xyz_fixed :
            The coordinates of the fixed (supported) nodes.
        structure :
            The structure that provides the free and fixed connectivity matrices.

        Returns
        -------
        residual :
            The term ``Cf.T @ diag(q) @ Cb @ xyz_fixed`` subtracted from the loads
            to form the right-hand side of the FDM system.
        """
        c_free = structure.connectivity_free
        c_fixed = structure.connectivity_fixed

        return c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    def residual_free_matrix(
        self,
        params: EquilibriumParametersState,
        xyz_free: Float[Array, "nodes_free 3"],
        structure: EquilibriumStructure,
    ) -> Float[Array, "nodes_free 3"]:
        """
        Compute the residual forces at the free nodes for a given geometry.

        Parameters
        ----------
        params :
            The force densities, fixed node coordinates, and load state.
        xyz_free :
            The current coordinates of the free nodes.
        structure :
            The structure that provides the connectivity matrices.

        Returns
        -------
        residuals :
            The out-of-balance force ``K @ xyz_free - P`` at each free node, zero
            at equilibrium.
        """
        # Unpack parameters
        q, xyz_fixed, load_state = params

        # Calculate stiffness matrix
        K = self.stiffness_matrix(q, structure)

        # Calculate load matrix
        P = self.load_xyz_matrix(params, xyz_free, structure)

        # Residual function
        residuals = K @ xyz_free - P

        return residuals


# ==========================================================================
# Sparse equilibrium model
# ==========================================================================


class EquilibriumModelSparse(EquilibriumModel):
    """
    An FDM model that solves the equilibrium system with a sparse linear solver.

    Notes
    -----
    Identical to [EquilibriumModel][jax_fdm.equilibrium.models.EquilibriumModel]
    except the stiffness matrix is assembled in sparse format and the linear
    system is solved with a sparse solver, which scales to larger structures.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearsolve_fn = spsolve

    @staticmethod
    def stiffness_matrix(
        q: Float[Array, "edges"],
        structure: EquilibriumStructureSparse,
    ) -> Float[CSC, "nodes_free nodes_free"]:
        """
        Assemble the force density stiffness matrix in sparse format.

        Parameters
        ----------
        q :
            The force densities of the edges.
        structure :
            The sparse structure that provides the sparse index array, diagonal
            indices, and the node-edge incidence used for the diagonal.

        Returns
        -------
        stiffness :
            The stiffness matrix of the free nodes, stored in sparse format.

        Notes
        -----
        The off-diagonal entries are the negated force densities gathered through
        the precomputed index array; the diagonal holds the per-node sum of
        incident force densities.
        """
        # shorthands
        index_array = structure.index_array
        diag_indices = structure.diag_indices
        diags = structure.diags

        nondiags_data = -q[index_array.data - 1]
        args = (nondiags_data, index_array.indices, index_array.indptr)

        K = CSC(args, shape=index_array.shape)

        # sum of force densities for each node
        # diag_fd = diags.T @ q  # for diags as CSC matrix
        diag_fd = diags @ q  # for diags as BCSR matrix
        K.data = K.data.at[diag_indices].set(diag_fd)

        return K
