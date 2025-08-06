"""
A collection of evolutionary optimizers.
"""
from time import perf_counter

from jax import vmap

from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

from jax_fdm.optimization.optimizers import Optimizer


# ==========================================================================
# Optimizers
# ==========================================================================


class EvolutionaryOptimizer(Optimizer):
    """
    An optimizer based on evolutionary principles.
    """
    def solve(self, opt_problem):
        """
        Solve an optimization problem by minimizing a loss function.
        """
        print(f"Optimization with {self.name} started...")
        start_time = perf_counter()
        res_q = self._minimize(opt_problem)
        end_time = perf_counter()

        loss_fn = opt_problem["func"]
        loss_val = loss_fn(res_q.x)

        print(f"Message: {res_q.message}")
        print(f"Final loss in {res_q.nit} iterations: {loss_val:.4}")
        print(f"Optimization elapsed time: {end_time() - start_time} seconds")

        return res_q.x


class DifferentialEvolution(EvolutionaryOptimizer):
    """
    The a differential evolution optimizer with box constraints.
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
        fun = opt_problem["fun"]

        def func(x):
            return fun(x)[0]

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


class DualAnnealing(EvolutionaryOptimizer):
    """
    The a dual annealing optimizer with box constraints.
    """
    def __init__(self, no_local_search=True, seed=None, display=False, **kwargs):
        super().__init__(name="DualAnnealing", disp=display, **kwargs)
        self.no_local_search = no_local_search
        self.seed = seed

    def _minimize(self, opt_problem):
        """
        Scipy backend method to minimize a loss function.
        """
        fun = opt_problem["fun"]

        def func(x, *args, **kwargs):
            return fun(x)[0]

        opt_problem["func"] = func
        opt_problem["no_local_search"] = self.no_local_search
        opt_problem["maxiter"] = opt_problem["options"]["maxiter"]
        opt_problem["args"] = (None, )

        del opt_problem["tol"]
        del opt_problem["fun"]
        del opt_problem["jac"]
        del opt_problem["hess"]
        del opt_problem["constraints"]
        del opt_problem["method"]
        del opt_problem["options"]

        return dual_annealing(**opt_problem)
