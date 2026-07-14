"""
A collection of scipy-powered, gradient-free optimizers.
"""
from collections.abc import Callable
from time import perf_counter
from typing import Any

import jax
from jax import jit

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import LoadState
from jax_fdm.losses import Loss
from jax_fdm.optimization.optimizers import Optimizer
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import Parameter
from jax_fdm.parameters import ParameterManager

# ==========================================================================
# Optimizers
# ==========================================================================

class GradientFreeOptimizer(Optimizer):
    """
    An optimizer based on evolutionary principles.
    """
    def problem(self,
                model: EquilibriumModel,
                structure: EquilibriumStructure,
                network: FDNetwork | FDMesh,
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
        # TODO: Merge this method with that of the upstream Optimizer() to avoid function duplication
        if not parameters:
            parameters = [EdgeForceDensityParameter(edge) for edge in network.edges()]

        self.pm = ParameterManager(model, parameters, structure, network)
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
        loads = LoadState.from_datastructure(network)
        self.loads_static = loads.edges, loads.faces

        # closure over static parameters
        def loss_fn(x: jax.Array) -> jax.Array | float:
            return self.loss(x, loss, model, structure)

        if jit_fn:
            loss_fn = jit(loss_fn)

        print("Warming up the pressure cooker...")
        start_time = perf_counter()
        loss_val = loss_fn(x)
        print(f"\tLoss warmup time: {(perf_counter() - start_time):.4} seconds")
        print(f"\tInitial loss value: {loss_val:.4}")

        # optimization options
        options = self.options(extra={"maxiter": maxiter})

        opt_kwargs = {"fun": loss_fn,
                      "jac": False,
                      "hess": None,
                      "method": self.name,
                      "x0": x,
                      "tol": tol,
                      "bounds": bounds,
                      "callback": callback,
                      "options": options
                      }

        return opt_kwargs

    def solve(self, opt_problem: dict[str, Any]) -> jax.Array:
        """
        Solve an optimization problem by minimizing a loss function.
        """
        print(f"Optimization with {self.name} started...")
        start_time = perf_counter()
        res_q = self._minimize(opt_problem)
        end_time = perf_counter()

        loss_fn = opt_problem.get("func")
        if not loss_fn:
            loss_fn = opt_problem["fun"]

        loss_val = loss_fn(res_q.x)

        print(f"Message: {res_q.message}")
        print(f"Final loss in {res_q.nit} iterations: {loss_val:.4} and {res_q.nfev} function evaluations")
        print(f"Optimization elapsed time: {end_time - start_time} seconds")

        return res_q.x


class Powell(GradientFreeOptimizer):
    """
    The modified Powell algorithm for gradient-free optimization with box constraints.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(name="Powell", disp=0, **kwargs)  # pyright: ignore[reportArgumentType]  # disp is declared as bool but scipy accepts an int verbosity level too; 0 is falsy and behaves like False here


class NelderMead(GradientFreeOptimizer):
    """
    The Nelder-Mead gradient-free optimizer with box constraints.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(name="Nelder-Mead", disp=0, **kwargs)  # pyright: ignore[reportArgumentType]  # disp is declared as bool but scipy accepts an int verbosity level too; 0 is falsy and behaves like False here
