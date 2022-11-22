from functools import partial

import jax.numpy as jnp

from jax import jit
from jax import vjp
from jax import jacfwd

from jax_fdm.optimization.optimizers import SecondOrderOptimizer
from jax_fdm.optimization.optimizers import ConstrainedOptimizer

try:
    from cyipopt import minimize_ipopt
except (ImportError, ModuleNotFoundError):
    pass


# ==========================================================================
# Constrained optimizer
# ==========================================================================

class IPOPT(ConstrainedOptimizer, SecondOrderOptimizer):
    """
    Interior Point Optimizer (Ipopt) for large-scale, gradient-based nonlinear optimization

    The optimizer supports box bounds on the optimization parameters, and equality and inequality constraints.

    Notes
    -----
    Ipopt expects that the loss and constraint functions are twice differentiable.
    """
    def __init__(self, acc_tol=1e-9, disp=1, **kwargs):
        super().__init__(name="IPOPT", disp=disp, **kwargs)
        self.acceptable_tol = acc_tol

    def _minimize(self, opt_problem):
        """
        Cyipopt backend method to minimize a loss function.
        """
        opt_problem["options"]["acceptable_tol"] = self.acceptable_tol

        return minimize_ipopt(**opt_problem)

    def constraint_eq(self, params_opt, constraint, model):
        """
        A wrapper function for an equality constraint on the parameters.
        """
        return self.constraint_ineq_low(params_opt, constraint, model)

    def constraint_ineq_up(self, params_opt, constraint, model):
        """
        A wrapper function for an inequality constraint on an upper bound of the parameters.
        """
        return constraint.bound_up - self.constraint(params_opt, constraint, model)

    def constraint_ineq_low(self, params_opt, constraint, model):
        """
        A wrapper function for an inequality constraint on a lower bound of the parameters.
        """
        return self.constraint(params_opt, constraint, model) - constraint.bound_low

    @staticmethod
    def hvp(x, v, f):
        """
        Calculate the product of the second derivatives of the vector-valued constraint function and a vector of Lagrange multipliers.
        """
        def _vjp(s):
            _, vjp_fun = vjp(f, s)
            return vjp_fun(v)[0]

        return jacfwd(_vjp)(x)

    def parameters_bounds(self):
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization parameters.
        """
        return list(zip(self.pm.bounds_low, self.pm.bounds_up))

    def constraints(self, constraints, model, params_opt):
        """
        Returns the defined constraints in a format amenable to `cyipopt`.
        """
        if not constraints:
            return []

        print(f"Constraints: {len(constraints)}")
        constraints = self.collect_constraints(constraints)
        print(f"\tConstraint colections: {len(constraints)}")

        clist = []
        for constraint in constraints:

            # initialize constraint
            constraint.init(model)

            # select constraint functions
            funs = []
            if jnp.allclose(constraint.bound_low, constraint.bound_up):
                ctype = "eq"
                funs.append(self.constraint_eq)
                print("\tAdding one equality constraint")
            else:
                ctype = "ineq"
                if not jnp.allclose(constraint.bound_low, -jnp.inf):
                    print("\tAdding bound low inequality constraint")
                    funs.append(self.constraint_ineq_low)
                if not jnp.allclose(constraint.bound_up, jnp.inf):
                    print("\tAdding bound high inequality constraint")
                    funs.append(self.constraint_ineq_up)
            if len(funs) == 0:
                print("\tNo constraints were processed. Check the constraint bounds!")
                return clist

            # format constraint functions in a dictionary
            for fun in funs:

                cdict = {}

                cfun = partial(fun, constraint=constraint, model=model)
                jac = jit(jacfwd(cfun))
                hvp = jit(partial(self.hvp, f=cfun))

                # warm start
                c = cfun(params_opt)
                _ = jac(params_opt)
                _ = hvp(params_opt, jnp.ones_like(c))

                # store
                cdict["type"] = ctype
                cdict["fun"] = cfun
                cdict["jac"] = jac
                cdict["hess"] = hvp

                clist.append(cdict)

        return clist
