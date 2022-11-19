from time import time
from timeit import timeit
from timeit import Timer

from jax_fdm.optimization.optimizers import SecondOrderOptimizer
from jax_fdm.optimization.optimizers import ConstrainedOptimizer

from functools import partial

from jax import jit
from jax import jacfwd, jacrev, hessian, jvp, grad, vmap

import jax.numpy as jnp

from scipy.optimize import BFGS

from cyipopt import minimize_ipopt

from jax.tree_util import Partial

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
    def __init__(self, acc_tol=1e-9, disp=5, **kwargs):
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
        """
        return self.constraint_ineq_low(params_opt, constraint, model)

    def constraint_ineq_up(self, params_opt, constraint, model):
        """
        """
        return constraint.bound_up - self.constraint(params_opt, constraint, model)

    def constraint_ineq_low(self, params_opt, constraint, model):
        """
        """
        return self.constraint(params_opt, constraint, model) - constraint.bound_low

    def constraints(self, constraints, model, params_opt):
        """
        Returns the defined constraints in a format amenable to `cyipopt`.
        """
        if not constraints:
            return []

        print(f"Constraints: {len(constraints)}")
        constraints = self.collect_constraints(constraints)
        print(f"Constraint colections: {len(constraints)}")

        clist = []
        for constraint in constraints:

            # initialize constraint
            constraint.init(model)

            # fun = partial(constraint, model=model)
            # cfuns = [partial(_constraint_eq, constraint=constraint, model=model)]
            cfuns = []

            # equality constraint
            # if constraint.bound_low == constraint.bound_up:
            #     ctype = "eq"
            #     print("Adding one equality constraint", constraint.bound_low)
            #     fun = partial(self.constraint_eq, constraint=constraint, model=model)
            #     _ = fun(params_opt)  # warm start
            #     cfuns.append(fun)

            # else:
            #     ctype = "ineq"
            #     if constraint.bound_low != -jnp.inf:
            #         print("Adding lower bound constraint", constraint.bound_low)
            #         fun = partial(self.constraint_ineq_low, constraint=constraint, model=model)
            #         _ = fun(params_opt)  # warm start
            #         cfuns.append(fun)
            #     if constraint.bound_up != jnp.inf:
            #         print("Adding upper bound constraint", constraint.bound_up)
            #         fun = partial(self.constraint_ineq_up, constraint=constraint, model=model)
            #         _ = fun(params_opt)  # warm start
            #         cfuns.append(fun)

            # ctype = "eq"
            # print("Adding one equality constraint")
            # fun = partial(self.constraint_eq, constraint=constraint, model=model)
            # _ = fun(params_opt)  # warm start
            # cfuns.append(fun)

            ctype = "ineq"
            for method in (self.constraint_ineq_low, self.constraint_ineq_up):
                cfun = partial(method, constraint=constraint, model=model)
                cfuns.append(cfun)

            for cfun in cfuns:

                cdict = {}

                jac = jit(jacfwd(cfun))

                # hessvp = jit(lambda x, v: self.hvp(cfun, (x, ), (v, )))  # hessian vector-product
                # hess = hessian(partial(self.dot_xv, cfun=cfun))
                # hess = jit(grad(lambda x, v: jnp.vdot(grad(cfun)(x), v))(x))
                # hess = hessian(cfun)
                hvp = jit(Partial(self.hvp, f=cfun))
                # hvp = jit(Partial(self.hvp2, f=cfun))

                # hess = jit(lambda x, v: jvp(grad(cfun), (x, ), (v, )))
                # hess = jit(partial(self.hvp2, f=cfun))

                c = cfun(params_opt)  # warm start
                j = jac(params_opt)  # warm start
                h = hvp(params_opt, jnp.ones_like(c))  # warm start
                assert h.shape == (params_opt.size, params_opt.size), f"We want: {(params_opt.size, params_opt.size)}"

                print(f"Shape of params opt: {params_opt.shape}")
                print(f"Shape of cfun: {c.shape}")
                print(f"Shape of jac: {j.shape}")
                print(f"Shape of hess VP in constraint: {h.shape}")

                n = 2
                times = []
                for i in range(n):
                    start_time = time()
                    _ = hvp(params_opt, jnp.ones_like(c))
                    times.append(time() - start_time)
                    print(time() - start_time)
                time_avg = sum(times) / n
                print(f"Hessian VO avg ex time: {time_avg:.4} seconds")

                # raise

                cdict["type"] = ctype
                cdict["fun"] = cfun
                cdict["jac"] = jac
                cdict["hess"] = hvp

                clist.append(cdict)

        return clist

    @staticmethod
    def hvp(x, v, f):
        h = hessian(f)(x)
        print(f"Hessian shape: {h.shape}")
        return jnp.tensordot(v, h, axes=1)

    @staticmethod
    def hvp2(primals, tangents, f):
        return jvp(jacfwd(f), (primals, ), (tangents, ))[1]
        # return jvp(grad(f), (primals, ), (tangents, ))[1]

    def parameters_bounds(self):
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization parameters.
        """
        return list(zip(self.pm.bounds_low, self.pm.bounds_up))
