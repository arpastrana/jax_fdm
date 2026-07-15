"""
A gradient-based optimizer.
"""
from collections.abc import Callable
from itertools import groupby
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
from jax_fdm.optimization import Collection
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import Parameter
from jax_fdm.parameters import ParameterManager

if TYPE_CHECKING:
    # deferred to avoid a runtime circular import (jax_fdm.goals imports back
    # into itself through goals/helpers.py); only needed for annotations.
    from jax_fdm.goals import Goal

# ==========================================================================
# Optimizer
# ==========================================================================

class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, name: str, disp: bool = True, **kwargs: Any):
        self.name = name
        self.disp = disp
        self.pm: ParameterManager | None = None
        self.loads_static: tuple[Float[Array, "edges 3"] | float, Float[Array, "faces 3"] | float] | None = None
        self.result: OptimizeResult | None = None

    def constraints(self, constraints: list[Any], model: EquilibriumModel, params_opt: Float[Array, "parameters"]) -> None:
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if constraints:
            print(f"\nWarning! {self.name} does not support constraints. I am ignoring them.")

    def gradient(self, loss: Callable) -> Callable:
        """
        Compute the gradient function of a loss function.
        """
        return jit(grad(loss, argnums=0))

    def hessian(self, loss: Callable) -> Callable | None:
        """
        Compute the hessian function of a loss function.
        """
        pass

# ==========================================================================
# Loss
# ==========================================================================

    def loss(self, params_opt: Float[Array, "parameters"], loss: Loss, model: EquilibriumModel, structure: EquilibriumStructure) -> Float[Array, ""]:
        """
        The wrapper loss.
        """
        params = self.parameters_fdm(params_opt)

        return loss(params, model, structure)

# ==========================================================================
# Goals
# ==========================================================================

    def goals(self, loss: Loss, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Pre-process the goals in the loss function to accelerate computations.
        """
        for term in loss.terms_error:
            goal_collections = self.collect_goals(term.goals)
            for goal_collection in goal_collections:
                goal_collection.init(model, structure)
            term.collections = goal_collections

# ==========================================================================
# Minimization
# ==========================================================================

    def problem(self,
                model: EquilibriumModel,
                structure: EquilibriumStructure,
                datastructure: FDNetwork | FDMesh,
                loss: Loss,
                parameters: list[Parameter] | None = None,
                constraints: list[Any] | None = None,
                maxiter: int = 100,
                tol: float = 1e-6,
                callback: Callable | None = None,
                jit_fn: bool = True) -> dict[str, Any]:
        """
        Set up an optimization problem.
        """
        # optimization parameters
        if not parameters:
            parameters = [EdgeForceDensityParameter(edge) for edge in datastructure.edges()]

        self.pm = ParameterManager(model, parameters, structure, datastructure)
        x = self.parameters_value()

        # message
        print(f"\n***Constrained form finding***\nParameters: {len(x)} \tGoals: {loss.number_of_goals()}")

        # parameter bounds
        bounds = self.parameters_bounds()

        assert x.size == self.pm.bounds_low.size
        assert x.size == self.pm.bounds_up.size

        # build goal collections
        self.goals(loss, model, structure)
        print(f"\tGoal collections: {loss.number_of_collections()}\n\tRegularizers: {loss.number_of_regularizers()}")

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
        print(f"\tLoss and grad warmup time: {(perf_counter() - start_time):.4} seconds")
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
        constraints = constraints or []
        if constraints:
            start_time = perf_counter()
            constraints = self.constraints(constraints, model, structure, x)  # pyright: ignore[reportCallIssue]  # base Optimizer.constraints() takes (constraints, model, params_opt); only ConstrainedOptimizer subclasses accept the extra `structure` arg used here
            print(f"\tConstraints warmup time: {(perf_counter() - start_time):.4} seconds")

        # optimization options
        options = self.options(extra={"maxiter": maxiter})

        opt_kwargs = {"fun": loss_and_grad_fn,
                      "jac": True,
                      "hess": hessian_fn,
                      "method": self.name,
                      "x0": x,
                      "tol": tol,
                      "bounds": bounds,
                      "constraints": constraints,
                      "callback": callback,
                      "options": options}

        return opt_kwargs

    def options(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Assemble a dictionary with method-specific optimization options.
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

    def solve(self, opt_problem: dict[str, Any]) -> Float[Array, "parameters"]:
        """
        Solve an optimization problem by minimizing a loss function.
        """
        # call callback with initial parameters
        callback = opt_problem.get("callback")
        if callback is not None:
            callback(opt_problem["x0"])

        print(f"Optimization with {self.name} started...")
        start_time = perf_counter()

        # minimize
        res_q = self._minimize(opt_problem)
        self.result = res_q
        loss_and_grad_fn = opt_problem["fun"]
        loss_val, grad_val = loss_and_grad_fn(res_q.x)

        print(f"Message: {res_q.message}")
        print(f"Final gradient norm: {jnp.linalg.norm(grad_val):.4}")
        print(f"Final loss in {res_q.nit} iterations: {loss_val:.4} and {res_q.nfev} function evaluations")
        print(f"Optimization elapsed time: {perf_counter() - start_time} seconds")

        return res_q.x

    @staticmethod
    def _minimize(opt_problem: dict[str, Any]) -> OptimizeResult:
        """
        Scipy backend method to minimize a loss function.
        """
        return minimize(**opt_problem)

# ==========================================================================
# Parameters
# ==========================================================================

    def parameters_bounds(self) -> Bounds:
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization parameters.
        """
        return Bounds(lb=self.pm.bounds_low, ub=self.pm.bounds_up)  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]  # self.pm is Optional by declaration but populated by problem() before this is called; bounds_low/up are ndarrays, Bounds also accepts array-likes despite the float-only stub

    def parameters_value(self) -> Float[Array, "parameters"]:
        """
        Return a flat array with the value of the optimization parameters.
        """
        return self.pm.parameters_value  # pyright: ignore[reportOptionalMemberAccess]  # self.pm is Optional by declaration but populated by problem() before this is called

    def parameters_fdm(self, params_opt: Float[Array, "parameters"]) -> EquilibriumParametersState:
        """
        Reconstruct the force density parameters from the optimization parameters.
        """
        params = self.pm.parameters_fdm(params_opt)  # pyright: ignore[reportOptionalMemberAccess]  # self.pm is Optional by declaration but populated by problem() before this is called

        q, xyz_fixed, loads_nodes = params
        loads_edges, loads_faces = self.loads_static  # pyright: ignore[reportGeneralTypeIssues]  # self.loads_static is Optional by declaration but populated by problem() before this is called

        loads = LoadState(nodes=loads_nodes,
                          edges=loads_edges,
                          faces=loads_faces)

        return EquilibriumParametersState(q=q,
                                          xyz_fixed=xyz_fixed,
                                          loads=loads)

# ==========================================================================
# Goal collections
# ==========================================================================

    @staticmethod
    def collect_goals(goals: list["Goal"]) -> list[Collection]:
        """
        Convert a list of goals into a list of goal collections.
        """
        goals_collectable = []
        goals_uncollectable = []

        for goal in goals:
            if goal.is_collectible:
                goals_collectable.append(goal)
            else:
                goals_uncollectable.append(goal)

        collections = []

        if goals_collectable:
            goals_sorted = sorted(goals_collectable, key=lambda g: type(g).__name__)
            groups = groupby(goals_sorted, lambda g: type(g))

            for _, group in groups:
                group = list(group)
                collection = Collection(group)
                collections.append(collection)

        if goals_uncollectable:
            for goal in goals_uncollectable:
                collections.append(Collection([goal]))

        return collections
