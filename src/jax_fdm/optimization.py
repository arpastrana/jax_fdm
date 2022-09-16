"""
A gradient-based optimizer.
"""
from time import time

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import grad
from jax import jacobian

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

from compas.data import Data

from jax_fdm.equilibrium import EquilibriumModel


# ==========================================================================
# Optimizer
# ==========================================================================


class Optimizer():
    def __init__(self, name, disp=True, **kwargs):
        self.name = name
        self.disp = disp

    def constraints(self, constraints, model, q):
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if constraints:
            print(f"Warning! {self.__class__.__name__} does not support constraints. I am ignoring them.")
        return ()

    def minimize(self, network, loss, bounds=(None, None), constraints=[], maxiter=100, tol=1e-6, verbose=True, callback=None):
        # returns the optimization result: dataclass OptimizationResult
        """
        Minimize a loss function via some flavor of gradient descent.
        """
        name = self.name

        # array-ize parameters
        q = jnp.asarray(network.edges_forcedensities(), dtype=jnp.float64)

        # message
        num_goals = sum([len(term.goals) for term in loss.loss_terms])
        print(f"\n***Constrained form finding***\nParameters: {len(q)} \tGoals: {num_goals} \tConstraints: {len(constraints)}")

        # create an equilibrium model from a network
        model = EquilibriumModel(network)

        # loss matters
        loss = partial(loss, model=model)

        # warm up loss
        start_time = time()
        loss(q)
        print(f"Loss warmup time: {round(time() - start_time, 4)} seconds")

        # gradient of the loss function
        grad_loss = jit(grad(loss))  # grad w.r.t. first function argument

        # warm up grad loss
        start_time = time()
        grad_loss(q)
        print(f"Gradient warmup time: {round(time() - start_time, 4)} seconds")

        # TODO: parameter bounds
        # bounds makes a re-index from one count system to the other
        # bounds = optimization_bounds(model, bounds)
        lb, ub = bounds
        if lb is None:
            lb = -jnp.inf
        if ub is None:
            ub = +jnp.inf

        bounds = Bounds(lb=lb, ub=ub)

        # TODO: support for scipy non-linear constraints
        start_time = time()
        constraints = self.constraints(constraints, model, q)
        print(f"Constraints warmup time: {round(time() - start_time, 4)} seconds")

        if verbose:
            print(f"Optimization with {self.name} started...")

        # scipy optimization
        start_time = time()

        # minimize
        res_q = minimize(fun=loss,
                         jac=grad_loss,
                         method=name,
                         x0=q,
                         tol=tol,
                         bounds=bounds,
                         constraints=constraints,
                         callback=callback,
                         options={"maxiter": maxiter, "disp": self.disp})
        # print out
        if verbose:
            print(res_q.message)
            print(f"Final loss in {res_q.nit} iterations: {res_q.fun}")
            print(f"Optimization elapsed time: {time() - start_time} seconds")

        return res_q.x

# ==========================================================================
# Optimizers
# ==========================================================================


class ConstrainedOptimizer(Optimizer):
    """
    A gradient-based optimizer that handles constraints.
    """
    def constraints(self, constraints, model, q):
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if not constraints:
            return

        clist = []
        for constraint in constraints:
            fun = jit(partial(constraint, model=model))
            jac = jit(jacobian(fun))
            lb = constraint.bound_low
            ub = constraint.bound_up
            # warm start
            fun(q)
            jac(q)
            # store non linear constraint
            clist.append(NonlinearConstraint(fun=fun, jac=jac, lb=lb, ub=ub))

        return clist

# ==========================================================================
# Optimizers
# ==========================================================================


class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="BFGS", **kwargs)


class TrustRegionConstrained(ConstrainedOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-constr", **kwargs)


class SLSQP(ConstrainedOptimizer):
    """
    The sequential least-squares programming optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="SLSQP", **kwargs)

    def constraints_dictionary(self, constraints, model):

        def _constraint_eq(q, constraint, model):
            return constraint.bound_up - constraint(q, model)

        def _constraint_ineq(q, constraint, model):
            return constraint(q, model) - constraint.bound_low

        if not constraints:
            return

        clist = []
        for constraint in constraints:

            # fun = partial(constraint, model=model)
            type = "eq"
            cfuns = [partial(_constraint_eq, constraint=constraint, model=model)]

            if constraint.bound_low != constraint.bound_up:
                type = "ineq"
                cfuns.append(partial(_constraint_ineq, constraint=constraint, model=model))

            for cfun in cfuns:
                cdict = dict()
                cdict["type"] = type
                cdict["fun"] = cfun
                cdict["jac"] = jacobian(cfun)
                clist.append(cdict)

        return clist

# ==========================================================================
# Recorder
# ==========================================================================


class OptimizationRecorder(Data):
    def __init__(self):
        self.history = []

    def record(self, value):
        self.history.append(value)

    def __call__(self, q, *args, **kwargs):
        self.record(q)

    @property
    def data(self):
        data = dict()
        data["history"] = self.history
        return data

    @data.setter
    def data(self, data):
        self.history = data["history"]
