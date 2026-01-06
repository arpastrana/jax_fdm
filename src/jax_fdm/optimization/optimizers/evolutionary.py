"""
A collection of evolutionary optimizers.
"""
from jax import vmap

from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

from jax_fdm.optimization.optimizers import GradientFreeOptimizer


# ==========================================================================
# Optimizers
# ==========================================================================

class DifferentialEvolution(GradientFreeOptimizer):
    """
    The a differential evolution optimizer with box constraints.

    This algorithm has stochastic components, so mind the seed for reproducibility.
    """
    def __init__(self, popsize=20, vectorized=False, num_workers=1, seed=43, display=False, **kwargs):
        super().__init__(name="DifferentialEvolution", disp=display, **kwargs)
        self.popsize = popsize
        self.vectorized = vectorized
        self.num_workers = num_workers
        self.seed = seed

    def _minimize(self, opt_problem):
        """
        Scipy backend method to minimize a loss function.
        """
        func = opt_problem["fun"]

        def func_vmap(x):
            result = vmap(func, in_axes=(1))(x)
            return result

        _args = None
        opt_problem["func"] = func
        if self.vectorized:
            opt_problem["func"] = func_vmap

        opt_problem["vectorized"] = self.vectorized
        opt_problem["polish"] = False
        opt_problem["seed"] = self.seed

        opt_problem["popsize"] = self.popsize
        opt_problem["maxiter"] = opt_problem["options"]["maxiter"]
        opt_problem["disp"] = opt_problem["options"]["disp"]
        opt_problem["args"] = _args
        opt_problem["updating"] = "deferred" if self.vectorized else "immediate"
        opt_problem["workers"] = self.num_workers

        del opt_problem["fun"]
        del opt_problem["jac"]
        del opt_problem["hess"]
        del opt_problem["method"]
        del opt_problem["options"]

        return differential_evolution(**opt_problem)


class DualAnnealing(GradientFreeOptimizer):
    """
    The a dual annealing optimizer with box constraints.

    This algorithm has stochastic components, so mind the seed for reproducibility.
    """
    def __init__(self, no_local_search=True, seed=42, display=False, **kwargs):
        super().__init__(name="DualAnnealing", disp=display, **kwargs)
        self.no_local_search = no_local_search
        self.seed = seed

    def _minimize(self, opt_problem):
        """
        Scipy backend method to minimize a loss function.
        """
        fun = opt_problem["fun"]

        def func(x, *args, **kwargs):
            return fun(x)

        opt_problem["func"] = func
        opt_problem["no_local_search"] = self.no_local_search
        opt_problem["maxiter"] = opt_problem["options"]["maxiter"]
        opt_problem["args"] = (None, )
        opt_problem["seed"] = self.seed

        del opt_problem["tol"]
        del opt_problem["fun"]
        del opt_problem["jac"]
        del opt_problem["hess"]
        del opt_problem["method"]
        del opt_problem["options"]

        return dual_annealing(**opt_problem)
