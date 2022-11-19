from time import time
from timeit import timeit
from timeit import Timer

from jax_fdm.optimization.optimizers import SecondOrderOptimizer
from jax_fdm.optimization.optimizers import ConstrainedOptimizer

from functools import partial

from jax import jit
from jax import jacfwd, jacrev, hessian, jvp, grad, vmap, vjp

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

            for fun in funs:

                cdict = {}

                cfun = partial(fun, constraint=constraint, model=model)
                jac = jit(jacfwd(cfun))
                hvp = jit(Partial(self.hvp, f=cfun))

                # warm start
                c = cfun(params_opt)
                j = jac(params_opt)
                h = hvp(params_opt, jnp.ones_like(c))


                # hessvp = jit(lambda x, v: self.hvp(cfun, (x, ), (v, )))  # hessian vector-product
                # hess = hessian(partial(self.dot_xv, cfun=cfun))
                # hess = jit(grad(lambda x, v: jnp.vdot(grad(cfun)(x), v))(x))
                # hess = hessian(cfun)
                # hvp = jit(Partial(self.hvp3, f=cfun))
                # hess = jit(jacfwd(jacfwd(cfun)))

                # hess = jit(lambda x, v: jvp(grad(cfun), (x, ), (v, )))
                # hess = jit(partial(self.hvp2, f=cfun))

                # hs = hess(params_opt)

                # print(f"Shape of params opt: {params_opt.shape}")
                # print(f"Shape of cfun: {c.shape}")
                # print(f"Shape of jac: {j.shape}")
                # print(f"Shape of hessian: {hs.shape}")
                # print(f"Shape of hess VP in constraint: {h.shape}")

                # h2 = hvp2(params_opt, jnp.ones_like(c))  # warm start
                # assert jnp.allclose(h, h2), f"\n{h}\nvs.\n{h2}"
                # assert
                # h.shape == (params_opt.size, params_opt.size), f"We want: {(params_opt.size, params_opt.size)}"

                # n = 3
                # times = []
                # for i in range(n):
                #     start_time = time()
                #     _ = jac(params_opt)
                #     times.append(time() - start_time)
                #     print(time() - start_time)
                # time_avg = sum(times) / n
                # print(f"Jacobian avg ex time: {time_avg:.4} seconds")

                # times = []
                # for i in range(n):
                #     start_time = time()
                #     _ = hess(params_opt)
                #     times.append(time() - start_time)
                #     print(time() - start_time)
                # time_avg = sum(times) / n
                # print(f"Hessian avg ex time: {time_avg:.4} seconds")

                # times = []
                # for i in range(n):
                #     start_time = time()
                #     _ = hvp(params_opt, jnp.ones_like(c))
                #     times.append(time() - start_time)
                #     print(time() - start_time)
                # time_avg = sum(times) / n
                # print(f"Hessian vector product avg ex time: {time_avg:.4} seconds")

                # raise
                cdict["type"] = ctype
                cdict["fun"] = cfun
                cdict["jac"] = jac
                cdict["hess"] = hvp

                clist.append(cdict)

        return clist

    @staticmethod
    def hvp(x, v, f):
        # h = hessian(f)(x)
        h = jacfwd(jacfwd(f))(x)
        # print(f"Hessian shape: {h.shape}")
        return jnp.tensordot(v, h, axes=1)

    @staticmethod
    def hvp(x, v, f):
        # h = hessian(f)(x)
        h = jacfwd(jacfwd(f))(x)
        return jnp.einsum("ijk,i->jk", h, v)

    @staticmethod
    def hvp4(x, v, f):
        return jacfwd(jvp(f, (x,), (v, ))[1])

    @staticmethod
    def hvp2(primals, tangents, f):
        # _, vjp_fun = vjp(grad(f), primals)
        # return vjp_fun(tangents)
        return jvp(jacfwd(f), (primals, ), (tangents, ))[1]
        # return jvp(grad(f), (primals, ), (tangents, ))[1]

    def parameters_bounds(self):
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization parameters.
        """
        return list(zip(self.pm.bounds_low, self.pm.bounds_up))
