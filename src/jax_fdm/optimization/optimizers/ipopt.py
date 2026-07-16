from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING
from typing import Any

import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vjp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm import has_backend
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.optimization import collect_constraints
from jax_fdm.optimization.optimizers import ConstrainedOptimizer
from jax_fdm.optimization.optimizers import OptProblem
from jax_fdm.optimization.optimizers import SecondOrderOptimizer

if TYPE_CHECKING:
    # Annotation-only import: pulling jax_fdm.constraints at runtime would form a
    # cycle (constraints -> equilibrium -> optimization).
    from jax_fdm.constraints import Constraint

if has_backend("cyipopt"):
    # cyipopt is an optional (ipopt extra) dependency, gated by has_backend above.
    from cyipopt import minimize_ipopt  # pyright: ignore[reportMissingImports]


# ==========================================================================
# Constrained optimizer
# ==========================================================================


class IPOPT(ConstrainedOptimizer, SecondOrderOptimizer):
    """
    Interior Point Optimizer (Ipopt) for large-scale, gradient-based nonlinear
    optimization

    The optimizer supports box bounds on the optimization parameters, and equality
    and inequality constraints.

    Notes
    -----
    Ipopt expects that the loss and constraint functions are twice differentiable.
    """

    name = "IPOPT"

    def __init__(self, acc_tol: float = 1e-9, **kwargs: Any):
        super().__init__(**kwargs)
        self.acceptable_tol = acc_tol

    def _minimize(self, opt_problem: OptProblem) -> Any:
        """
        Cyipopt backend method to minimize a loss function.
        """
        opt_problem.options["acceptable_tol"] = self.acceptable_tol
        # Ipopt expects an integer print level rather than a boolean flag.
        opt_problem.options["disp"] = int(self.disp)

        return minimize_ipopt(**opt_problem.to_kwargs())

    def constraint_eq(
        self,
        params_opt: Float[Array, "parameters"],
        constraint: "Constraint",
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "constraints"]:
        """
        A wrapper function for an equality constraint on the parameters.
        """
        return self.constraint_ineq_low(params_opt, constraint, model, structure)

    def constraint_ineq_up(
        self,
        params_opt: Float[Array, "parameters"],
        constraint: "Constraint",
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "constraints"]:
        """
        A wrapper function for an inequality constraint on an upper bound of the
        parameters.
        """
        return constraint.bound_up - self.constraint(
            params_opt,
            constraint,
            model,
            structure,
        )

    def constraint_ineq_low(
        self,
        params_opt: Float[Array, "parameters"],
        constraint: "Constraint",
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "constraints"]:
        """
        A wrapper function for an inequality constraint on a lower bound of the
        parameters.
        """
        return (
            self.constraint(params_opt, constraint, model, structure)
            - constraint.bound_low
        )

    @staticmethod
    def hvp(
        x: Float[Array, "parameters"],
        v: Float[Array, "constraints"],
        f: Callable,
    ) -> Float[Array, "parameters"]:
        """
        Calculate the product of the second derivatives of the vector-valued
        constraint function and a vector of Lagrange multipliers.
        """

        def _vjp(s: Float[Array, "parameters"]) -> Float[Array, "parameters"]:
            _, vjp_fun = vjp(f, s)
            return vjp_fun(v)[0]

        return jacfwd(_vjp)(x)

    def parameters_bounds(self) -> list[tuple[float, float]]:
        """
        Return a tuple of arrays with the upper and the lower bounds of optimization
        parameters.
        """
        return list(zip(self.pm.bounds_low, self.pm.bounds_up))

    def constraints(
        self,
        constraints: list["Constraint"],
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params_opt: Float[Array, "parameters"],
    ) -> list[dict[str, Any]]:
        """
        Returns the defined constraints in a format amenable to `cyipopt`.
        """
        if not constraints:
            return []

        print(f"Constraints: {len(constraints)}")
        collections = collect_constraints(constraints)
        print(f"\tConstraint colections: {len(collections)}")

        clist = []
        for constraint in collections:
            # initialize constraint
            constraint.init(model, structure)

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

                cfun = partial(
                    fun,
                    constraint=constraint,
                    model=model,
                    structure=structure,
                )
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
