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
    The interior point optimizer (Ipopt) for large-scale nonlinear optimization.

    Parameters
    ----------
    acc_tol :
        The acceptable convergence tolerance Ipopt may settle for.

    Notes
    -----
    Supports box bounds and both equality and inequality constraints. Ipopt expects
    the loss and constraint functions to be twice differentiable, so this optimizer
    supplies hessians and constraint hessian-vector products.
    """

    name = "IPOPT"

    def __init__(self, acc_tol: float = 1e-9, **kwargs: Any):
        super().__init__(**kwargs)
        self.acceptable_tol = acc_tol

    def _minimize(self, opt_problem: OptProblem) -> Any:
        """
        Dispatch the problem to the cyipopt backend.

        Parameters
        ----------
        opt_problem :
            The problem to minimize.

        Returns
        -------
        result :
            The cyipopt optimization result.
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
        The residual of an equality constraint, zero when satisfied.

        Parameters
        ----------
        params_opt :
            The flat optimization parameter vector.
        constraint :
            The constraint to evaluate.
        model :
            The equilibrium model.
        structure :
            The structure that provides the connectivity.

        Returns
        -------
        residual :
            The constrained quantity minus its (shared) bound.
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
        The slack against a constraint's upper bound, non-negative when satisfied.

        Parameters
        ----------
        params_opt :
            The flat optimization parameter vector.
        constraint :
            The constraint to evaluate.
        model :
            The equilibrium model.
        structure :
            The structure that provides the connectivity.

        Returns
        -------
        slack :
            The upper bound minus the constrained quantity.
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
        The slack against a constraint's lower bound, non-negative when satisfied.

        Parameters
        ----------
        params_opt :
            The flat optimization parameter vector.
        constraint :
            The constraint to evaluate.
        model :
            The equilibrium model.
        structure :
            The structure that provides the connectivity.

        Returns
        -------
        slack :
            The constrained quantity minus its lower bound.
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
        The constraint hessian contracted with the Lagrange multipliers.

        Parameters
        ----------
        x :
            The point to evaluate the hessian-vector product at.
        v :
            The vector of Lagrange multipliers, one per constraint component.
        f :
            The vector-valued constraint function.

        Returns
        -------
        hvp :
            The sum of each constraint component's hessian scaled by its multiplier,
            applied at ``x``.
        """

        def _vjp(s: Float[Array, "parameters"]) -> Float[Array, "parameters"]:
            _, vjp_fun = vjp(f, s)
            return vjp_fun(v)[0]

        return jacfwd(_vjp)(x)

    def parameters_bounds(self) -> list[tuple[float, float]]:
        """
        Return the parameter bounds as (low, high) pairs for cyipopt.

        Returns
        -------
        bounds :
            One (lower, upper) bound pair per optimization parameter.
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
        Convert constraints into cyipopt constraint dictionaries.

        Parameters
        ----------
        constraints :
            The constraints to convert.
        model :
            The equilibrium model.
        structure :
            The structure the constraints are defined on.
        params_opt :
            The initial optimization parameters, used to warm up the jitted
            functions.

        Returns
        -------
        constraints :
            One dictionary per constraint side, each with a value function, its
            Jacobian, and its hessian-vector product.

        Notes
        -----
        An equal lower and upper bound yields a single equality constraint;
        otherwise each finite bound becomes its own one-sided inequality.
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
