from collections.abc import Callable
from typing import Any

from jaxtyping import Array
from jaxtyping import Float
from optimistix import ImplicitAdjoint
from optimistix import RecursiveCheckpointAdjoint

from jax_fdm.equilibrium.solvers.types import SolverIterParams

__all__ = ["solver_optimistix"]


def solver_optimistix(
    solver_cls: Callable,
    routine_fn: Callable,
    fn: Callable,
    a: SolverIterParams,
    x_init: Float[Array, "..."],
    solver_config: dict[str, Any],
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[Array, "..."]:
    """
    Solve ``fn(a, x)`` with an Optimistix solver and routine.

    Parameters
    ----------
    solver_cls :
        The Optimistix solver class to instantiate.
    routine_fn :
        The Optimistix routine that runs the solve, e.g. ``least_squares`` or
        ``root_find``.
    fn :
        The function to iterate upon.
    a :
        The function parameters.
    x_init :
        The initial guess for the solution vector.
    solver_config :
        The configuration options of the solver, read for ``tmax``, ``eta``,
        ``verbose``, and ``implicit_diff``.
    solver_kwargs :
        Extra keyword arguments forwarded to the solver class. If None, no extras
        are passed.

    Returns
    -------
    x_star :
        The solution vector.

    Notes
    -----
    Selects implicit-adjoint differentiation when ``implicit_diff`` is set, and
    recursive-checkpoint adjoint otherwise.
    """
    tmax = solver_config["tmax"]
    eta = solver_config["eta"]
    verbose = solver_config["verbose"]
    implicit_diff = solver_config["implicit_diff"]

    if implicit_diff:
        adjoint = ImplicitAdjoint()
    else:
        adjoint = RecursiveCheckpointAdjoint()

    if solver_kwargs is None:
        solver_kwargs = {}

    if verbose:
        stats = {"loss", "step_size"}
        verbose = frozenset(stats)
        solver_kwargs["verbose"] = verbose

    def fn_swapped(x, a):
        return fn(a, x)

    solver = solver_cls(
        rtol=eta,
        atol=eta,
        **solver_kwargs,
    )

    solution = routine_fn(
        fn=fn_swapped,
        solver=solver,
        y0=x_init,
        args=a,
        has_aux=False,
        max_steps=tmax,
        throw=False,
        adjoint=adjoint,
    )

    return solution.value
