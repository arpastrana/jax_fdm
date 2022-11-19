"""
A gradient-based optimizer.
"""
from time import time
from itertools import groupby
from functools import partial

from jax import jit
from jax import grad

from scipy.optimize import minimize
from scipy.optimize import Bounds

from jax_fdm.parameters import ParameterManager
from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.optimization import Collection


# ==========================================================================
# Optimizer
# ==========================================================================

class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, name, disp=True):
        self.name = name
        self.disp = disp
        self.pm = None

    def constraints(self, constraints, model, params_opt):
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if constraints:
            print(f"Warning! {self.name} does not support constraints. I am ignoring them.")

    def gradient(self, loss):
        """
        Compute the gradient function of a loss function.
        """
        return jit(grad(loss, argnums=0))

    def hessian(self, loss):
        """
        Compute the hessian function of a loss function.
        """
        pass

# ==========================================================================
# Loss
# ==========================================================================

    @partial(jit, static_argnums=(0, 2, 3))
    def loss(self, params_opt, loss, model):
        """
        The wrapper loss.
        """
        q, xyz_fixed, loads = self.parameters_fdm(params_opt)

        return loss(q, xyz_fixed, loads, model)

# ==========================================================================
# Goals
# ==========================================================================

    def goals(self, loss, model):
        """
        Pre-process the goals in the loss function to accelerate computations.
        """
        for term in loss.terms_error:
            goal_collections = self.collect_goals(term.goals)
            for goal_collection in goal_collections:
                goal_collection.init(model)
            term.collections = goal_collections

# ==========================================================================
# Minimization
# ==========================================================================

    def problem(self,
                model,
                loss,
                parameters=None,
                constraints=None,
                maxiter=100,
                tol=1e-6,
                callback=None):
        """
        Set up an optimization problem.
        """
        # optimization parameters
        if not parameters:
            parameters = [EdgeForceDensityParameter(edge) for edge in model.structure.edges]

        self.pm = ParameterManager(model, parameters)
        x = self.parameters_opt()

        # message
        print(f"\n***Constrained form finding***\nParameters: {len(x)} \tGoals: {loss.number_of_goals()}")

        # parameter bounds
        bounds = self.parameters_bounds()

        assert x.size == self.pm.bounds_low.size
        assert x.size == self.pm.bounds_up.size

        # build goal collections
        self.goals(loss, model)
        print(f"\tGoal collections: {loss.number_of_collections()}")

        # loss matters
        loss = partial(self.loss, loss=loss, model=model)

        print("Warming up the pressure cooker...")
        start_time = time()
        _ = loss(x)
        print(f"\tLoss warmup time: {(time() - start_time):.4} seconds")

        # gradient of the loss function
        grad_loss = self.gradient(loss)  # w.r.t. first function argument
        start_time = time()
        _ = grad_loss(x)
        print(f"\tGradient warmup time: {(time() - start_time):.4} seconds")

        # gradient of the loss function
        hessian_loss = self.hessian(loss)  # w.r.t. first function argument
        if hessian_loss:
            start_time = time()
            _ = hessian_loss(x)
            print(f"\tHessian warmup time: {(time() - start_time):.4} seconds")

        # constraints
        constraints = constraints or []
        if constraints:
            start_time = time()
            constraints = self.constraints(constraints, model, x)
            print(f"\tConstraints warmup time: {round(time() - start_time, 4)} seconds")

        opt_kwargs = {"fun": loss,
                      "jac": grad_loss,
                      "hess": hessian_loss,
                      "method": self.name,
                      "x0": x,
                      "tol": tol,
                      "bounds": bounds,
                      "constraints": constraints,
                      "callback": callback,
                      "options": {"maxiter": maxiter, "disp": self.disp}}

        return opt_kwargs

    def solve(self, opt_problem):
        """
        Solve an optimization problem by minimizing a loss function via gradient descent.
        """
        print(f"Optimization with {self.name} started...")
        start_time = time()

        # minimize
        res_q = self._minimize(opt_problem)

        print(res_q.message)
        print(f"Final loss in {res_q.nit} iterations: {res_q.fun}")
        print(f"Optimization elapsed time: {time() - start_time} seconds")

        return res_q.x

    @staticmethod
    def _minimize(opt_problem):
        """
        Scipy backend method to minimize a loss function.
        """
        return minimize(**opt_problem)

# ==========================================================================
# Parameters
# ==========================================================================

    def parameters_bounds(self):
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization parameters.
        """
        return Bounds(lb=self.pm.bounds_low, ub=self.pm.bounds_up)

    def parameters_opt(self):
        """
        Return a flat array with the optimization parameters.
        """
        return self.pm.parameters_opt

    def parameters_frozen(self):
        """
        Return a flat array with the parameters that must stay constant during optimization.
        """
        return self.pm.parameters_frozen

    def parameters_fdm(self, params_opt):
        """
        Reconstruct the force density parameters from the optimization parameters.
        """
        return self.pm.parameters_fdm(params_opt)

# ==========================================================================
# Goal collections
# ==========================================================================

    @staticmethod
    def collect_goals(goals):
        """
        Convert a list of goals into a list of goal collections.
        """
        goals = sorted(goals, key=lambda g: type(g).__name__)
        groups = groupby(goals, lambda g: type(g))

        collections = []
        for _, group in groups:
            collection = Collection(list(group))
            collections.append(collection)

        return collections
