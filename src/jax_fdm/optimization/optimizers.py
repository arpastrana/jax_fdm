"""
A gradient-based optimizer.
"""
from time import time

from itertools import groupby
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import grad
from jax import jacobian

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumModel

from jax_fdm.goals import goals_reindex
from jax_fdm.optimization import Collection

from jax_fdm.losses import Regularizer


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

    def constraints(self, constraints, model, q):
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if constraints:
            print(f"Warning! {self.name} does not support constraints. I am ignoring them.")
        return ()

    def minimize(self, network, loss, bounds=(None, None), constraints=[], maxiter=100, tol=1e-6, verbose=True, callback=None):
        """
        Minimize a loss function via some flavor of gradient descent.
        """
        # array-ize parameters
        q = jnp.asarray(network.edges_forcedensities(), dtype=DTYPE_JAX)

        # message
        num_goals = 0
        for term in loss.terms:
            if isinstance(term, Regularizer):
                continue
            for goal in term.goals:
                num_goals += 1
        print(f"\n***Constrained form finding***\nParameters: {len(q)} \tGoals: {num_goals} \tConstraints: {len(constraints)}")

        # create an equilibrium model from a network
        model = EquilibriumModel(network)

        # TODO: gather goal collections for acceleration
        for term in loss.terms:

            # skip regularizers
            if isinstance(term, Regularizer):
                continue

            goals = term.goals

            # sort goals by class name
            goals = sorted(goals, key=lambda g: type(g).__name__)
            groups = groupby(goals, lambda g: type(g))

            collections = []
            for key, goal_group in groups:
                gc = Collection(list(goal_group))
                collections.append(gc)

            # collections.extend(uncollectibles)
            term.goals = collections

        # reindex goals
        for term in loss.terms:
            if isinstance(term, Regularizer):
                continue
            goals_reindex(term.goals, model)

        # loss matters
        loss = partial(loss, model=model)

        print("Warming up the pressure cooker...")

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
                         method=self.name,
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
