"""SciPy-backed gradient-free optimizers."""

from collections.abc import Callable
from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING

from jax import jit
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import LoadState
from jax_fdm.losses import Loss
from jax_fdm.optimization.optimizers import Optimizer
from jax_fdm.optimization.optimizers import OptProblem
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import Parameter
from jax_fdm.parameters import ParameterManager

if TYPE_CHECKING:
    # Annotation-only import: pulling jax_fdm.constraints at runtime would form a
    # cycle (constraints -> equilibrium -> optimization).
    from jax_fdm.constraints import Constraint

# ==========================================================================
# Optimizers
# ==========================================================================


class GradientFreeOptimizer(Optimizer):
    """
    The base class for optimizers that minimize without gradients.

    Notes
    -----
    Overrides `problem` to build an objective without a gradient or hessian,
    so the loss is only jitted for its value. Constraints are not supported.
    """

    def problem(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        datastructure: FDNetwork | FDMesh,
        loss: Loss,
        parameters: Sequence[Parameter] | None = None,
        constraints: Sequence["Constraint"] | None = None,
        maxiter: int = 100,
        tol: float = 1e-6,
        callback: Callable | None = None,
        jit_fn: bool = True,
    ) -> OptProblem:
        """
        Assemble a gradient-free optimization problem.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure that provides the connectivity.
        datastructure :
            The network or mesh being optimized.
        loss :
            The loss function to minimize.
        parameters :
            The optimization parameters. If None, every edge force density is used.
        constraints :
            Ignored; gradient-free optimizers do not support constraints.
        maxiter :
            The maximum number of optimizer iterations.
        tol :
            The convergence tolerance.
        callback :
            A function invoked once per iteration.
        jit_fn :
            Whether to just-in-time compile the objective.

        Returns
        -------
        problem :
            The assembled optimization problem, without a gradient or hessian.
        """
        # TODO: Merge this method with that of the upstream Optimizer() to avoid
        # function duplication
        if not parameters:
            parameters = [
                EdgeForceDensityParameter(edge) for edge in datastructure.edges()
            ]

        self.pm = ParameterManager(model, parameters, structure, datastructure)
        x = self.parameters_value()

        # message
        print(
            f"\n***Constrained form finding***\n"
            f"Parameters: {len(x)} \tGoals: {loss.number_of_goals()}",
        )

        # parameter bounds
        bounds = self.parameters_bounds()

        assert x.size == self.pm.bounds_low.size
        assert x.size == self.pm.bounds_up.size

        # build goal collections
        self.goals(loss, model, structure)
        print(
            f"\tGoal collections: {loss.number_of_collections()}\n"
            f"\tRegularizers: {loss.number_of_regularizers()}",
        )

        # load matters
        loads = LoadState.from_datastructure(datastructure)
        self.loads_static = loads.edges, loads.faces

        # closure over static parameters
        def loss_fn(x: Float[Array, "parameters"]) -> Float[Array, ""]:
            return self.loss(x, loss, model, structure)

        if jit_fn:
            loss_fn = jit(loss_fn)

        print("Warming up the pressure cooker...")
        start_time = perf_counter()
        loss_val = loss_fn(x)
        print(f"\tLoss warmup time: {(perf_counter() - start_time):.4} seconds")
        print(f"\tInitial loss value: {loss_val:.4}")

        # optimization options
        options = self.options(extra={"maxiter": maxiter})

        return OptProblem(
            fun=loss_fn,
            jac=False,
            hess=None,
            method=self.name,
            x0=x,
            tol=tol,
            bounds=bounds,
            callback=callback,
            options=options,
        )

    def solve(self, opt_problem: OptProblem) -> Float[Array, "parameters"]:
        """
        Minimize an assembled gradient-free problem.

        Parameters
        ----------
        opt_problem :
            The problem to minimize.

        Returns
        -------
        params_opt :
            The optimized parameter vector.
        """
        print(f"Optimization with {self.name} started...")
        start_time = perf_counter()
        res_q = self._minimize(opt_problem)
        end_time = perf_counter()

        loss_val = opt_problem.fun(res_q.x)

        print(f"Message: {res_q.message}")
        print(
            f"Final loss in {res_q.nit} iterations: {loss_val:.4} and "
            f"{res_q.nfev} function evaluations",
        )
        print(f"Optimization elapsed time: {end_time - start_time} seconds")

        return res_q.x


class Powell(GradientFreeOptimizer):
    """
    The modified Powell algorithm for gradient-free optimization with box constraints.
    """

    name = "Powell"


class NelderMead(GradientFreeOptimizer):
    """
    The Nelder-Mead gradient-free optimizer with box constraints.
    """

    name = "Nelder-Mead"
