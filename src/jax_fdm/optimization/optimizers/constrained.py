"""A gradient-based optimizer that handles equality and inequality constraints."""

from collections.abc import Sequence
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
from jax_fdm.optimization.collections import collect_constraints
from jax_fdm.optimization.optimizers.optimizer import Optimizer

if TYPE_CHECKING:
    # Annotation-only import: pulling jax_fdm.constraints at runtime would form a
    # cycle (constraints -> equilibrium -> optimization).
    from jax_fdm.constraints import Constraint

__all__ = ["ConstrainedOptimizer"]

# ==========================================================================
# Constrained optimizer
# ========== ===============================================================


class ConstrainedOptimizer(Optimizer):
    """
    A gradient-based optimizer that handles constraints.
    """

    def constraints(
        self,
        constraints: Sequence["Constraint"],
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params_opt: Float[Array, "parameters"],
    ) -> list[Any] | None:
        """
        Convert constraints into SciPy ``NonlinearConstraint`` objects.

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
            constraint and its Jacobian.

        Returns
        -------
        constraints :
            The SciPy constraints, or None when there are none.

        Notes
        -----
        Each constraint carries a jitted value function and a forward-mode Jacobian.
        Subclasses may return a different container: ``IPOPT`` returns cyipopt
        constraint dictionaries instead.
        """
        if not constraints:
            return

        print(f"Constraints: {len(constraints)}")
        collections = collect_constraints(constraints)
        print(f"\tConstraint colections: {len(collections)}")

        clist = []
        for constraint in collections:
            # Nested def (not partial): jit(partial(...)) can resolve to the
            # decorator overload, so calls appear to expect a Callable.
            def fun(
                params: Float[Array, "parameters"],
            ) -> Float[Array, "constraints"]:
                return self.constraint(params, constraint, model, structure)

            fun_jit = jit(fun)
            jac = jit(jacfwd(fun))

            lb = constraint.bound_low
            ub = constraint.bound_up

            # warm start
            fun_jit(params_opt)
            jac(params_opt)

            # store non linear constraint
            # scipy's NonlinearConstraint stub types `jac` as a literal string
            # mode selector; a callable Jacobian is also valid per scipy docs
            clist.append(NonlinearConstraint(fun=fun_jit, jac=jac, lb=lb, ub=ub))  # pyright: ignore[reportArgumentType]

        return clist

    def constraint(
        self,
        params_opt: Float[Array, "parameters"],
        constraint: "Constraint",
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "constraints"]:
        """
        Evaluate a constraint from a flat optimization parameter vector.

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
        values :
            The constrained quantity for each element.
        """
        params: EquilibriumParametersState = self.parameters_fdm(params_opt)

        return constraint(params, model, structure)
