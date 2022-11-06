"""
A gradient-based optimizer.
"""
from time import time
from itertools import groupby
from functools import partial

from jax import jit
from jax import grad
from jax import jacobian
from jax import hessian

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

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

    def constraints(self, constraints, model, q, xyz_fixed, loads):
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
        return

    def minimize(self, model, loss, parameters=None, constraints=None, maxiter=100, tol=1e-6, callback=None):
        """
        Minimize a loss function via some flavor of gradient descent.
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
        for term in loss.terms_error:
            goal_collections = self.collect_goals(term.goals)
            for goal_collection in goal_collections:
                goal_collection.init(model)
            term.collections = goal_collections
        print(f"Goal collections: {loss.number_of_collections()}")

        # loss matters
        loss = partial(self.loss, loss=loss, model=model)

        print("Warming up the pressure cooker...")
        start_time = time()
        loss(x)
        print(f"Loss warmup time: {round(time() - start_time, 4)} seconds")

        # gradient of the loss function
        grad_loss = self.gradient(loss)  # w.r.t. first function argument
        start_time = time()
        grad_loss(x)
        print(f"Gradient warmup time: {round(time() - start_time, 4)} seconds")

        # gradient of the loss function
        hessian_loss = self.hessian(loss)  # w.r.t. first function argument
        if hessian_loss:
            start_time = time()
            hessian_loss(x)
            print(f"Hessian warmup time: {round(time() - start_time, 4)} seconds")

        # constraints
        if constraints:
            start_time = time()
            constraints = self.constraints(constraints, model, x)
            print(f"Constraints warmup time: {round(time() - start_time, 4)} seconds")

        # scipy optimization
        print(f"Optimization with {self.name} started...")
        start_time = time()

        # minimize
        res_q = minimize(fun=loss,
                         jac=grad_loss,
                         hess=hessian_loss,
                         method=self.name,
                         x0=x,  # q
                         tol=tol,
                         bounds=bounds,
                         constraints=constraints,
                         callback=callback,
                         options={"maxiter": maxiter, "disp": self.disp})

        print(res_q.message)
        print(f"Final loss in {res_q.nit} iterations: {res_q.fun}")
        print(f"Optimization elapsed time: {time() - start_time} seconds")

        return res_q.x

# ==========================================================================
# Goals
# ==========================================================================

    def collect_goals(self, goals):
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

    @partial(jit, static_argnums=0)
    def parameters_fdm(self, params_opt):
        """
        Reconstruct the force density parameters from the optimization parameters.
        """
        return self.pm.parameters_fdm(params_opt)


# ==========================================================================
# Constrained optimizer
# ==========================================================================

class ConstrainedOptimizer(Optimizer):
    """
    A gradient-based optimizer that handles constraints.
    """
    def collect_constraints(self, constraints):
        """
        Convert a list of constraints into a list of constraint collections.
        """
        constraints = sorted(constraints, key=lambda g: type(g).__name__)
        groups = groupby(constraints, lambda g: type(g))

        collections = []
        for _, group in groups:
            collection = Collection(list(group))
            collections.append(collection)

        return collections

    def constraints(self, constraints, model, params_opt):
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if not constraints:
            return

        print(f"Constraints: {len(constraints)}")
        constraints = self.collect_constraints(constraints)
        print(f"Constraint colections: {len(constraints)}")

        clist = []
        for constraint in constraints:
            # initialize constraint
            constraint.init(model)

            # gather information for scipy constraint
            fun = partial(self.constraint, constraint=constraint, model=model)

            jac = jit(jacobian(fun))
            lb = constraint.bound_low
            ub = constraint.bound_up

            # warm start
            fun(params_opt)
            jac(params_opt)

            # store non linear constraint
            clist.append(NonlinearConstraint(fun=fun, jac=jac, lb=lb, ub=ub))

        return clist

    @partial(jit, static_argnums=(0, 2, 3))
    def constraint(self, params_opt, constraint, model):
        """
        A wrapper around a constraint callable object.
        """
        q, xyz_fixed, loads = self.parameters_fdm(params_opt)

        return constraint(q, xyz_fixed, loads, model)


# ==========================================================================
# Second-order optimizer
# ==========================================================================

class SecondOrderOptimizer(Optimizer):
    """
    A gradient-based optimizer that uses the hessian to accelerate convergence.
    """
    def hessian(self, loss):
        """
        Compute the hessian function of a loss function.
        """
        return jit(hessian(loss, argnums=0))


# ==========================================================================
# Optimizers
# ==========================================================================

class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="BFGS", **kwargs)


class LBFGSB(Optimizer):
    """
    The limited-memory Boyd-Fletcher-Floyd-Shannon-Byrd optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="L-BFGS-B", disp=0, **kwargs)


class TrustRegionConstrained(ConstrainedOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-constr", **kwargs)


class NewtonCG(SecondOrderOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Newton-CG", **kwargs)


class SLSQP(ConstrainedOptimizer):
    """
    The sequential least-squares programming optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="SLSQP", **kwargs)
