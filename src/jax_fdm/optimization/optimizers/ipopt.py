from functools import partial

import jax.numpy as jnp

from jax import jit

from jax import jacfwd

from jax_fdm.optimization.optimizers import SecondOrderOptimizer
from jax_fdm.optimization.optimizers import ConstrainedOptimizer

try:
    from cyipopt import minimize_ipopt
except (ImportError, ModuleNotFoundError):
    pass

from jax import vmap, grad, jacrev
from jax import jvp, vjp


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
        Calculate the product of the Hessian of the constraint function and a vector of Lagrange multipliers.
        """
        hessian = jacfwd(jacfwd(f))(x)
        return jnp.einsum("ijk,i->jk", hessian, v)

    @staticmethod
    def vhvp(x, v, f):
        """
        Calculate the product of the Hessian of the constraint function and a vector of Lagrange multipliers.
        Vectorized.
        """
        print(f"x shape: {x.shape}, v shape: {v.shape}")

        # y, vjp_fun = vjp(f, x)
        # outs, = vmap(vjp_fun)(M)
        # return outs

        # y, vjp_fun = vjp(f, x)
        # outs, = vmap(vjp_fun)(jacfwd(x))

        # return vmap(jvp, in_axes=(0, None))(jacfwd(f), (x, ), (v, ))
        # jacobian = jacfwd(f)(x)
        thing = lambda s: jnp.dot(v, jacfwd(f)(s))
        return jacfwd(thing)(x)

        # one option
        # _jvp = lambda s: jvp(jacfwd(f), (x, ), (s, ))[1]
        # return vmap(_jvp, in_axes=(0, ))(v)

        # _, tangents = jvp(jacfwd(f), (jacobian[:, 1], ), (v, ))
        # _, vjp_fun = vjp(f, x)
        # out = vjp_fun(v)

        # return tangents
        # hessian = jacfwd(jacfwd(f))(x)
        # return jnp.einsum("ijk,i->jk", hessian, v)
        return

    @staticmethod
    def hvp2(x, v, f):
        """
        Calculate the product of the Hessian of the constraint function and a vector of Lagrange multipliers.
        Vectorized.
        """
        thing = lambda s: jnp.dot(v, jacfwd(f)(s))
        return jacfwd(thing)(x)

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

            for fun in funs:

                cdict = {}

                cfun = partial(fun, constraint=constraint, model=model)
                jac = jit(jacfwd(cfun))
                hvp = jit(partial(self.hvp, f=cfun))
                vhvp = jit(partial(self.vhvp, f=cfun))

                # warm start
                c = cfun(params_opt)
                _ = jac(params_opt)
                h = hvp(params_opt, jnp.ones_like(c))

                print("hvp shape",  h.shape)
                vh = vhvp(params_opt, jnp.ones_like(c))
                print()
                print("vhvp shape", vh.shape)
                assert jnp.allclose(h, vh)

                n = 5
                from time import time
                times = []
                for i in range(n):
                    start_time = time()
                    _ = hvp(params_opt, jnp.ones_like(c))
                    times.append(time() - start_time)
                    print(time() - start_time)
                time_avg = sum(times) / n
                print(f"Hessian vector product avg ex time: {time_avg:.4} seconds")

                times = []
                for i in range(n):
                    start_time = time()
                    _ = vhvp(params_opt, jnp.ones_like(c))
                    times.append(time() - start_time)
                    print(time() - start_time)
                time_avg = sum(times) / n
                print(f"Hessian vector product 2 avg ex time: {time_avg:.4} seconds")
                #
                raise


                # store
                cdict["type"] = ctype
                cdict["fun"] = cfun
                cdict["jac"] = jac
                cdict["hess"] = hvp

                clist.append(cdict)

        return clist
