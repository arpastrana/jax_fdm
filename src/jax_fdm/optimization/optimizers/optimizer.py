"""The base optimizer and the SciPy problem it assembles."""

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Any

import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import value_and_grad
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import Bounds
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import LoadState
from jax_fdm.losses import Loss
from jax_fdm.optimization import collect_goals
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import Parameter
from jax_fdm.parameters import ParameterManager

if TYPE_CHECKING:
    # Annotation-only import: pulling jax_fdm.constraints at runtime would form a
    # cycle (constraints -> equilibrium -> optimization).
    from jax_fdm.constraints import Constraint

__all__ = ["OptProblem", "Optimizer"]

# ==========================================================================
# Optimization problem
# ==========================================================================


@dataclass
class OptProblem:
    """
    The arguments that describe an optimization problem.

    This mirrors the signature of ``scipy.optimize.minimize``. The fields common
    to every backend are named explicitly; the gradient-free and evolutionary
    backends need a different keyword set (e.g. renaming ``fun`` to ``func`` and
    dropping ``jac``/``hess``), so each ``_minimize`` override reads the fields it
    needs and assembles its own keyword arguments rather than sharing one dict.
    """

    fun: Callable
    x0: Float[Array, "parameters"]
    method: str
    options: dict[str, Any]
    jac: bool | Callable = False
    hess: Callable | None = None
    tol: float | None = None
    bounds: Bounds | list[tuple[float, float]] | None = None
    constraints: list[Any] = field(default_factory=list)
    callback: Callable | None = None

    def to_kwargs(self) -> dict[str, Any]:
        """
        Expand the problem into keyword arguments for a SciPy backend.

        Returns
        -------
        kwargs :
            The problem fields as a keyword-argument dictionary.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ==========================================================================
# Optimizer
# ==========================================================================

LoadsStatic = tuple[Float[Array, "edges 3"] | float, Float[Array, "faces 3"] | float]


class Optimizer:
    """
    The base class for optimizers backed by ``scipy.optimize.minimize``.

    Parameters
    ----------
    disp :
        Whether the SciPy backend prints its own console output. Off by default
        since jax_fdm prints its own progress.

    Notes
    -----
    Subclasses set ``name`` to their SciPy method identity and override the
    gradient, hessian, constraint, or minimization hooks as needed. `problem`
    assembles the objective, its gradient, bounds, and goal collections into an
    `OptProblem`, which `solve` then minimizes.
    """

    name: str = ""

    def __init__(self, disp: bool = False, **kwargs: Any):
        # `name` is the fixed scipy method identity of each optimizer, so it is a
        # class attribute rather than a constructor argument. `disp` toggles the
        # backend's own console output; jax_fdm already prints its own progress,
        # so it stays quiet by default.
        self.disp = disp
        self._pm: ParameterManager | None = None
        self._loads_static: LoadsStatic | None = None
        self.result: OptimizeResult | None = None

    @property
    def pm(self) -> ParameterManager:
        """
        The parameter manager, set up by ``problem()`` before a solve.
        """
        if self._pm is None:
            raise RuntimeError("The parameter manager is unset; call problem() first.")
        return self._pm

    @pm.setter
    def pm(self, value: ParameterManager) -> None:
        self._pm = value

    @property
    def loads_static(self) -> LoadsStatic:
        """
        The static edge and face loads, set up by ``problem()`` before a solve.
        """
        if self._loads_static is None:
            raise RuntimeError("The static loads are unset; call problem() first.")
        return self._loads_static

    @loads_static.setter
    def loads_static(self, value: LoadsStatic) -> None:
        self._loads_static = value

    def constraints(
        self,
        constraints: Sequence["Constraint"],
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params_opt: Float[Array, "parameters"],
    ) -> list[Any] | None:
        """
        Convert constraints into the form the SciPy backend expects.

        Parameters
        ----------
        constraints :
            The constraints to convert.
        model :
            The equilibrium model.
        structure :
            The structure the constraints are defined on.
        params_opt :
            The initial optimization parameters.

        Returns
        -------
        constraints :
            The converted constraints. The base optimizer supports none, so it warns
            and returns None.
        """
        if constraints:
            print(
                f"\nWarning! {self.name} does not support constraints. "
                f"I am ignoring them.",
            )

    def gradient(self, loss: Callable) -> Callable:
        """
        Build the jitted gradient of a loss function.

        Parameters
        ----------
        loss :
            The loss function to differentiate.

        Returns
        -------
        gradient :
            The jitted gradient with respect to the optimization parameters.
        """
        return jit(grad(loss, argnums=0))

    def hessian(self, loss: Callable) -> Callable | None:
        """
        Build the hessian of a loss function.

        Parameters
        ----------
        loss :
            The loss function to differentiate twice.

        Returns
        -------
        hessian :
            The hessian function, or None. The base optimizer provides none;
            second-order optimizers override this.
        """
        pass

    # ==========================================================================
    # Loss
    # ==========================================================================

    def loss(
        self,
        params_opt: Float[Array, "parameters"],
        loss: Loss,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, ""]:
        """
        Evaluate the loss from a flat optimization parameter vector.

        Parameters
        ----------
        params_opt :
            The flat optimization parameter vector.
        loss :
            The loss function to evaluate.
        model :
            The equilibrium model.
        structure :
            The structure that provides the connectivity.

        Returns
        -------
        loss :
            The scalar loss value.

        Notes
        -----
        Expands the optimization vector into FDM parameters before calling the loss,
        so the optimizer can work in the reduced parameter space.
        """
        params = self.parameters_fdm(params_opt)

        return loss(params, model, structure)

    # ==========================================================================
    # Goals
    # ==========================================================================

    def goals(
        self,
        loss: Loss,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> None:
        """
        Collect and bind the loss's goals to a structure ahead of the solve.

        Parameters
        ----------
        loss :
            The loss function whose error terms hold the goals.
        model :
            The equilibrium model.
        structure :
            The structure the goals are bound to.

        Notes
        -----
        Goals are batched into collections and each collection is initialized once,
        so the per-iteration objective avoids re-resolving goal indices.
        """
        for term in loss.terms_error:
            goal_collections = collect_goals(term.goals)
            for goal_collection in goal_collections:
                goal_collection.init(model, structure)
            term.collections = goal_collections

    # ==========================================================================
    # Minimization
    # ==========================================================================

    def problem(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        datastructure: FDNetwork | FDMesh,
        loss: Loss,
        parameters: Sequence[Parameter] | None = None,
        constraints: Sequence["Constraint"] | None = None,
        maxiter: int = 100,
        tol: float = 1e-6,
        callback: Callable | None = None,
        jit_fn: bool = True,
    ) -> OptProblem:
        """
        Assemble an optimization problem from a model, loss, and parameters.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure that provides the connectivity.
        datastructure :
            The network or mesh being optimized.
        loss :
            The loss function to minimize.
        parameters :
            The optimization parameters. If None, every edge force density is used.
        constraints :
            The constraints to enforce. If None, the problem is unconstrained.
        maxiter :
            The maximum number of optimizer iterations.
        tol :
            The convergence tolerance.
        callback :
            A function invoked once per iteration.
        jit_fn :
            Whether to just-in-time compile the objective and hessian.

        Returns
        -------
        problem :
            The assembled optimization problem.

        Notes
        -----
        Warms up the compiled objective once and asserts the gradient is nan-free
        before returning, so compilation cost and gradient bugs surface up front.
        """
        # optimization parameters
        if not parameters:
            parameters = [
                EdgeForceDensityParameter(edge) for edge in datastructure.edges()
            ]

        self.pm = ParameterManager(model, parameters, structure, datastructure)
        x = self.parameters_value()

        # message
        print(
            f"\n***Constrained form finding***\n"
            f"Parameters: {len(x)} \tGoals: {loss.number_of_goals()}",
        )

        # parameter bounds
        bounds = self.parameters_bounds()

        assert x.size == self.pm.bounds_low.size
        assert x.size == self.pm.bounds_up.size

        # build goal collections
        self.goals(loss, model, structure)
        print(
            f"\tGoal collections: {loss.number_of_collections()}\n"
            f"\tRegularizers: {loss.number_of_regularizers()}",
        )

        # load matters
        loads = LoadState.from_datastructure(datastructure)
        self.loads_static = loads.edges, loads.faces

        # closure over static parameters
        def loss_fn(x: Float[Array, "parameters"]) -> Float[Array, ""]:
            return self.loss(x, loss, model, structure)

        loss_and_grad_fn = value_and_grad(loss_fn)
        if jit_fn:
            loss_and_grad_fn = jit(loss_and_grad_fn)

        print("Warming up the pressure cooker...")
        start_time = perf_counter()
        loss_val, grad_val = loss_and_grad_fn(x)
        print(
            f"\tLoss and grad warmup time: {(perf_counter() - start_time):.4} seconds",
        )
        print(f"\tInitial loss value: {loss_val:.4}")
        print(f"\tInitial gradient norm: {jnp.linalg.norm(grad_val):.4}")
        assert jnp.sum(jnp.isnan(grad_val)) == 0, "NaNs found in gradient calculation!"

        # hessian of the loss function
        # TODO: move to second-order optimizers
        hessian_fn = self.hessian(loss_fn)  # w.r.t. first function argument
        if hessian_fn:
            if jit_fn:
                hessian_fn = jit(hessian_fn)
            start_time = perf_counter()
            _ = hessian_fn(x)
            print(f"\tHessian warmup time: {(perf_counter() - start_time):.4} seconds")

        # constraints
        constraints = list(constraints or [])
        if constraints:
            start_time = perf_counter()
            constraints = self.constraints(constraints, model, structure, x) or []
            print(
                f"\tConstraints warmup time: "
                f"{(perf_counter() - start_time):.4} seconds",
            )

        # optimization options
        options = self.options(extra={"maxiter": maxiter})

        return OptProblem(
            fun=loss_and_grad_fn,
            jac=True,
            hess=hessian_fn,
            method=self.name,
            x0=x,
            tol=tol,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options=options,
        )

    def options(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Assemble the backend options dictionary.

        Parameters
        ----------
        extra :
            Extra options to merge in, such as ``maxiter``. None-valued entries are
            skipped.

        Returns
        -------
        options :
            The options dictionary, always carrying the ``disp`` flag.

        Raises
        ------
        ValueError
            If ``extra`` is given but is not a dictionary.
        """
        options = {"disp": self.disp}

        if extra is None:
            return options

        if not isinstance(extra, dict):
            raise ValueError("Extra options must be a dictionary!")

        for key, value in extra.items():
            if value is None:
                continue
            options[key] = value

        return options

    def solve(self, opt_problem: OptProblem) -> Float[Array, "parameters"]:
        """
        Minimize an assembled optimization problem.

        Parameters
        ----------
        opt_problem :
            The problem to minimize.

        Returns
        -------
        params_opt :
            The optimized parameter vector.

        Notes
        -----
        Also stores the full SciPy result on ``self.result`` and prints the final
        loss, gradient norm, and iteration counts.
        """
        # call callback with initial parameters
        if opt_problem.callback is not None:
            opt_problem.callback(opt_problem.x0)

        print(f"Optimization with {self.name} started...")
        start_time = perf_counter()

        # minimize
        res_q = self._minimize(opt_problem)
        self.result = res_q
        loss_and_grad_fn = opt_problem.fun
        loss_val, grad_val = loss_and_grad_fn(res_q.x)

        print(f"Message: {res_q.message}")
        print(f"Final gradient norm: {jnp.linalg.norm(grad_val):.4}")
        print(
            f"Final loss in {res_q.nit} iterations: {loss_val:.4} and "
            f"{res_q.nfev} function evaluations",
        )
        print(f"Optimization elapsed time: {perf_counter() - start_time} seconds")

        return res_q.x

    def _minimize(self, opt_problem: OptProblem) -> OptimizeResult:
        """
        Dispatch the problem to the SciPy minimizer.

        Parameters
        ----------
        opt_problem :
            The problem to minimize.

        Returns
        -------
        result :
            The raw SciPy optimization result.
        """
        return minimize(**opt_problem.to_kwargs())

    # ==========================================================================
    # Parameters
    # ==========================================================================

    def parameters_bounds(self) -> Bounds | list[tuple[float, float]]:
        """
        Return the lower and upper bounds of the optimization parameters.

        Most backends consume a scipy ``Bounds`` object; ``IPOPT`` overrides this
        to return a list of ``(low, high)`` pairs instead.
        """
        # bounds_low/up are ndarrays; Bounds also accepts array-likes despite
        # its float-only stub
        return Bounds(lb=self.pm.bounds_low, ub=self.pm.bounds_up)  # pyright: ignore[reportArgumentType]

    def parameters_value(self) -> Float[Array, "parameters"]:
        """
        Return the initial values of the optimization parameters.

        Returns
        -------
        values :
            The flat initial optimization parameter vector.
        """
        return self.pm.parameters_value

    def parameters_fdm(
        self,
        params_opt: Float[Array, "parameters"],
    ) -> EquilibriumParametersState:
        """
        Expand optimization parameters into a full FDM parameter state.

        Parameters
        ----------
        params_opt :
            The flat optimization parameter vector.

        Returns
        -------
        params_state :
            The force densities, fixed coordinates, and loads, with the static edge
            and face loads restored.
        """
        params = self.pm.parameters_fdm(params_opt)

        q, xyz_fixed, loads_nodes = params
        loads_edges, loads_faces = self.loads_static

        loads = LoadState(nodes=loads_nodes, edges=loads_edges, faces=loads_faces)

        return EquilibriumParametersState(q=q, xyz_fixed=xyz_fixed, loads=loads)
