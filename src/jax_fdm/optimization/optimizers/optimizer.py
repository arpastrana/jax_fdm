"""
A gradient-based optimizer.
"""
from time import perf_counter
from itertools import groupby
from functools import partial

import jax.numpy as jnp

from jax import jit
from jax import grad
from jax import value_and_grad

from scipy.optimize import minimize
from scipy.optimize import Bounds

from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import EquilibriumParametersState

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
        self.loads_static = None

    def constraints(self, constraints, model, params_opt):
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if constraints:
            print(f"\nWarning! {self.name} does not support constraints. I am ignoring them.")

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

    def loss(self, params_opt, loss, model, structure):
        """
        The wrapper loss.
        """
        params = self.parameters_fdm(params_opt)

        return loss(params, model, structure)

# ==========================================================================
# Goals
# ==========================================================================

    def goals(self, loss, model, structure):
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
        # optimization parameters
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

        loss_fn = partial(self.loss, loss=loss, model=model, structure=structure)
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

        # gradient of the loss function
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
            constraints = self.constraints(constraints, model, structure, x)
            print(f"\tConstraints warmup time: {(perf_counter() - start_time):.4} seconds")

        opt_kwargs = {"fun": loss_and_grad_fn,
                      "jac": True,
                      "hess": hessian_fn,
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
        start_time = perf_counter()

        # minimize
        res_q = self._minimize(opt_problem)
        loss_and_grad_fn = opt_problem["fun"]
        loss_val, grad_val = loss_and_grad_fn(res_q.x)

        print(res_q.message)
        print(f"Final gradient norm: {jnp.linalg.norm(grad_val):.4}")
        print(f"Final loss in {res_q.nit} iterations: {loss_val:.4}")
        print(f"Optimization elapsed time: {perf_counter() - start_time} seconds")

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

    def parameters_value(self):
        """
        Return a flat array with the value of the optimization parameters.
        """
        return self.pm.parameters_value

    def parameters_fdm(self, params_opt):
        """
        Reconstruct the force density parameters from the optimization parameters.
        """
        params = self.pm.parameters_fdm(params_opt)

        q, xyz_fixed, loads_nodes = params
        loads_edges, loads_faces = self.loads_static

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
    def collect_goals(goals):
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
