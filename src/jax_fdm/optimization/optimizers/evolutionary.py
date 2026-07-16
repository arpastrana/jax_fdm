"""
A collection of evolutionary optimizers.
"""
from typing import Any

from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

from jax_fdm.optimization.optimizers import GradientFreeOptimizer
from jax_fdm.optimization.optimizers import OptProblem

# ==========================================================================
# Optimizers
# ==========================================================================

class DifferentialEvolution(GradientFreeOptimizer):
    """
    The a differential evolution optimizer with box constraints.

    This algorithm has stochastic components, so mind the seed for reproducibility.
    """
    name = "DifferentialEvolution"

    def __init__(
        self,
        popsize: int = 20,
        vectorized: bool = False,
        num_workers: int = 1,
        seed: int = 43,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.popsize = popsize
        self.vectorized = vectorized
        self.num_workers = num_workers
        self.seed = seed

    def _minimize(self, opt_problem: OptProblem) -> OptimizeResult:
        """
        Scipy backend method to minimize a loss function.
        """
        func = opt_problem.fun

        def func_vmap(x: Float[Array, "parameters population"]) -> Float[Array, "population"]:
            result = vmap(func, in_axes=(1))(x)
            return result

        return differential_evolution(
            func=func_vmap if self.vectorized else func,
            x0=opt_problem.x0,
            tol=opt_problem.tol,
            bounds=opt_problem.bounds,
            callback=opt_problem.callback,
            vectorized=self.vectorized,
            polish=False,
            seed=self.seed,  # pyright: ignore[reportCallIssue]  # scipy 1.18's stub renamed `seed` to `rng`, but the installed runtime still accepts `seed`; the min supported scipy predates `rng`
            popsize=self.popsize,
            maxiter=opt_problem.options["maxiter"],
            disp=opt_problem.options["disp"],
            args=None,
            updating="deferred" if self.vectorized else "immediate",
            workers=self.num_workers,
        )


class DualAnnealing(GradientFreeOptimizer):
    """
    The a dual annealing optimizer with box constraints.

    This algorithm has stochastic components, so mind the seed for reproducibility.
    """
    name = "DualAnnealing"

    def __init__(
        self,
        no_local_search: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.no_local_search = no_local_search
        self.seed = seed

    def _minimize(self, opt_problem: OptProblem) -> OptimizeResult:
        """
        Scipy backend method to minimize a loss function.
        """
        fun = opt_problem.fun

        def func(x: Float[Array, "parameters"], *args: Any, **kwargs: Any) -> Float[Array, ""]:
            return fun(x)

        return dual_annealing(
            func=func,
            x0=opt_problem.x0,
            bounds=opt_problem.bounds,
            callback=opt_problem.callback,
            no_local_search=self.no_local_search,
            maxiter=opt_problem.options["maxiter"],
            args=(None, ),
            seed=self.seed,  # pyright: ignore[reportCallIssue]  # scipy 1.18's stub renamed `seed` to `rng`, but the installed runtime still accepts `seed`; the min supported scipy predates `rng`
        )
