"""
A gradient-based optimizer that deals with equality and inequality constraints.
"""
from functools import partial
from typing import TYPE_CHECKING
from typing import Any

from jax import jacfwd
from jax import jit
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import NonlinearConstraint

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.optimization import collect_constraints
from jax_fdm.optimization.optimizers import Optimizer

if TYPE_CHECKING:
    # Annotation-only import: pulling jax_fdm.constraints at runtime would form a
    # cycle (constraints -> equilibrium -> optimization).
    from jax_fdm.constraints import Constraint

# ==========================================================================
# Constrained optimizer
# ========== ===============================================================

class ConstrainedOptimizer(Optimizer):
    """
    A gradient-based optimizer that handles constraints.
    """
    def constraints(
        self,
        constraints: list["Constraint"],
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params_opt: Float[Array, "parameters"],
    ) -> list[Any] | None:
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.

        Subclasses may return a different container: `IPOPT` returns a list of
        cyipopt constraint dictionaries rather than scipy `NonlinearConstraint`s.
        """
        if not constraints:
            return

        print(f"Constraints: {len(constraints)}")
        collections = collect_constraints(constraints)
        print(f"\tConstraint colections: {len(collections)}")

        clist = []
        for constraint in collections:

            # initialize constraint
            constraint.init(model, structure)

            # gather information for scipy constraint
            fun = partial(self.constraint,
                          constraint=constraint,
                          model=model,
                          structure=structure)

            fun = jit(fun)
            jac = jit(jacfwd(fun))

            lb = constraint.bound_low
            ub = constraint.bound_up

            # warm start
            fun(params_opt)
            jac(params_opt)

            # store non linear constraint
            clist.append(NonlinearConstraint(fun=fun, jac=jac, lb=lb, ub=ub))  # pyright: ignore[reportArgumentType]  # scipy's NonlinearConstraint stub types `jac` as a literal string mode selector; a callable Jacobian is also valid per scipy docs

        return clist

    def constraint(
        self,
        params_opt: Float[Array, "parameters"],
        constraint: "Constraint",
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "constraints"]:
        """
        A wrapper around a constraint callable object.
        """
        params: EquilibriumParametersState = self.parameters_fdm(params_opt)

        return constraint(params, model, structure)
