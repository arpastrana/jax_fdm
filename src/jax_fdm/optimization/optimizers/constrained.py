"""
A gradient-based optimizer that deals with equality and inequality constraints.
"""
from functools import partial
from itertools import groupby

from jax import jacfwd
from jax import jit
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import NonlinearConstraint

from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.optimization import Collection
from jax_fdm.optimization.optimizers import Optimizer

# ==========================================================================
# Constrained optimizer
# ========== ===============================================================

class ConstrainedOptimizer(Optimizer):
    """
    A gradient-based optimizer that handles constraints.
    """
    def constraints(
        self,
        constraints: list[Constraint],
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params_opt: Float[Array, "parameters"],
    ) -> list[NonlinearConstraint] | None:
        """
        Returns the defined constraints in a format amenable to `scipy.minimize`.
        """
        if not constraints:
            return

        print(f"Constraints: {len(constraints)}")
        constraints = self.collect_constraints(constraints)  # pyright: ignore[reportAssignmentType]  # reused as a local for the resulting Collection list, shadowing the incoming list[Constraint] parameter type
        print(f"\tConstraint colections: {len(constraints)}")

        clist = []
        for constraint in constraints:

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
        constraint: Constraint,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "constraints"]:
        """
        A wrapper around a constraint callable object.
        """
        params: EquilibriumParametersState = self.parameters_fdm(params_opt)

        return constraint(params, model, structure)

    @staticmethod
    def collect_constraints(constraints: list[Constraint]) -> list[Collection]:
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
