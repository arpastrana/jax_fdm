"""
A collection of scipy-powered, gradient-free optimizers.
"""
from time import perf_counter

from jax import jit

from jax_fdm.equilibrium import LoadState

from jax_fdm.parameters import ParameterManager
from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.optimization.optimizers import Optimizer


# ==========================================================================
# Optimizers
# ==========================================================================

class GradientFreeOptimizer(Optimizer):
    """
    An optimizer based on evolutionary principles.
    """
    def problem(self,
                model,
                structure,
                network,
                loss,
                parameters=None,
                constraints=None,
                maxiter=100,
                tol=1e-6,
                callback=None,
                jit_fn=True):
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
        def loss_fn(x):
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
                      "constraints": None,
                      "callback": callback,
                      "options": options
                      }

        return opt_kwargs

    def solve(self, opt_problem):
        """
        Solve an optimization problem by minimizing a loss function.
        """
        print(f"Optimization with {self.name} started...")
        start_time = perf_counter()
        res_q = self._minimize(opt_problem)
        end_time = perf_counter()

        loss_fn = opt_problem.get("func", opt_problem["fun"])
        loss_val = loss_fn(res_q.x)

        print(f"Message: {res_q.message}")
        print(f"Final loss in {res_q.nit} iterations: {loss_val:.4}")
        print(f"Optimization elapsed time: {end_time - start_time} seconds")

        return res_q.x


class Powell(GradientFreeOptimizer):
    """
    The modified Powell algorithm for gradient-free optimization with box constraints.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Powell", disp=0, **kwargs)


class NelderMead(GradientFreeOptimizer):
    """
    The Nelder-Mead gradient-free optimizer with box constraints.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Nelder-Mead", disp=0, **kwargs)
