from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vjp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.optimization.optimizers import ConstrainedOptimizer
from jax_fdm.optimization.optimizers import SecondOrderOptimizer

try:
    from cyipopt import minimize_ipopt  # pyright: ignore[reportMissingImports]  # cyipopt is an optional (ipopt extra) dependency
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
    def __init__(self, acc_tol: float = 1e-9, disp: int = 1, **kwargs: Any):
        super().__init__(name="IPOPT", disp=disp, **kwargs)  # pyright: ignore[reportArgumentType]  # disp is declared as bool but scipy/cyipopt-style verbosity accepts an int level too
        self.acceptable_tol = acc_tol

    def _minimize(self, opt_problem: dict[str, Any]) -> Any:
        """
        Cyipopt backend method to minimize a loss function.
        """
        opt_problem["options"]["acceptable_tol"] = self.acceptable_tol

        return minimize_ipopt(**opt_problem)

    def constraint_eq(self, params_opt: Float[Array, "parameters"], constraint: Constraint, model: EquilibriumModel) -> Float[Array, "constraints"]:
        """
        A wrapper function for an equality constraint on the parameters.
        """
        return self.constraint_ineq_low(params_opt, constraint, model)

    def constraint_ineq_up(self, params_opt: Float[Array, "parameters"], constraint: Constraint, model: EquilibriumModel) -> Float[Array, "constraints"]:
        """
        A wrapper function for an inequality constraint on an upper bound of the parameters.
        """
        return constraint.bound_up - self.constraint(params_opt, constraint, model)  # pyright: ignore[reportCallIssue]  # ConstrainedOptimizer.constraint() also requires a `structure` arg; this IPOPT-specific 3-arg wrapper predates that signature

    def constraint_ineq_low(self, params_opt: Float[Array, "parameters"], constraint: Constraint, model: EquilibriumModel) -> Float[Array, "constraints"]:
        """
        A wrapper function for an inequality constraint on a lower bound of the parameters.
        """
        return self.constraint(params_opt, constraint, model) - constraint.bound_low  # pyright: ignore[reportCallIssue]  # ConstrainedOptimizer.constraint() also requires a `structure` arg; this IPOPT-specific 3-arg wrapper predates that signature

    @staticmethod
    def hvp(x: Float[Array, "parameters"], v: Float[Array, "constraints"], f: Callable) -> Float[Array, "parameters"]:
        """
        Calculate the product of the second derivatives of the vector-valued constraint function and a vector of Lagrange multipliers.
        """
        def _vjp(s: Float[Array, "parameters"]) -> Float[Array, "parameters"]:
            _, vjp_fun = vjp(f, s)
            return vjp_fun(v)[0]

        return jacfwd(_vjp)(x)

    def parameters_bounds(self) -> list[tuple[float, float]]:
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization parameters.
        """
        return list(zip(self.pm.bounds_low, self.pm.bounds_up))  # pyright: ignore[reportOptionalMemberAccess]  # self.pm is Optional by declaration but populated by problem() before this is called

    def constraints(self, constraints: list[Constraint], model: EquilibriumModel, params_opt: Float[Array, "parameters"]) -> list[dict[str, Any]]:
        """
        Returns the defined constraints in a format amenable to `cyipopt`.
        """
        if not constraints:
            return []

        print(f"Constraints: {len(constraints)}")
        constraints = self.collect_constraints(constraints)  # pyright: ignore[reportAssignmentType]  # reused as a local for the resulting Collection list, shadowing the incoming list[Constraint] parameter type
        print(f"\tConstraint colections: {len(constraints)}")

        clist = []
        for constraint in constraints:

            # initialize constraint
            constraint.init(model)  # pyright: ignore[reportCallIssue]  # base Constraint.init() takes (model, structure); IPOPT's collect_constraints path predates that signature and only ever calls it with model

            # select constraint functions
            funs = []
            if jnp.allclose(constraint.bound_low, constraint.bound_up):  # pyright: ignore[reportArgumentType]  # bound_low/bound_up are declared Optional but are always populated to a float/array by their setters before this point
                ctype = "eq"
                funs.append(self.constraint_eq)
                print("\tAdding one equality constraint")
            else:
                ctype = "ineq"
                if not jnp.allclose(constraint.bound_low, -jnp.inf):  # pyright: ignore[reportArgumentType]  # bound_low is declared Optional but always populated to a float/array by its setter before this point
                    print("\tAdding bound low inequality constraint")
                    funs.append(self.constraint_ineq_low)
                if not jnp.allclose(constraint.bound_up, jnp.inf):  # pyright: ignore[reportArgumentType]  # bound_up is declared Optional but always populated to a float/array by its setter before this point
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
