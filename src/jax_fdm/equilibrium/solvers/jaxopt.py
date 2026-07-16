from collections.abc import Callable
from typing import Any

from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium.solvers.types import SolverIterParams


def solver_jaxopt(
    solver_cls: Callable,
    fn: Callable,
    a: SolverIterParams,
    x_init: Float[Array, "..."],
    solver_config: dict[str, Any],
    solver_kwargs: dict[str, Any] | None = None,
) -> Float[Array, "..."]:
    """
    Find a fixed point of ``fn(a, x)`` with a JAXopt solver class.

    Parameters
    ----------
    solver_cls :
        The JAXopt solver class to instantiate.
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
        The solution vector at the fixed point.

    Notes
    -----
    When implicit differentiation is off the iteration is unrolled so reverse-mode
    autodiff can flow through: JAXopt otherwise uses a ``lax.while_loop``, which is
    not reverse-mode differentiable.
    """
    tmax = solver_config["tmax"]
    eta = solver_config["eta"]
    verbose = solver_config["verbose"]
    implicit_diff = solver_config["implicit_diff"]

    if solver_kwargs is None:
        solver_kwargs = {}

    # NOTE: Unroll python loop if solver config disables implicit diff
    # This enables reverse-mode AD to calculate gradients when implicit differentiation
    # is off because the solver uses lax.while_loop under the hood to not unroll
    # iterations but this type of while loop is not reverse-mode differentiable.
    unroll = False
    if not implicit_diff:
        unroll = True

    def fn_swapped(x, a):
        return fn(a, x)

    solver = solver_cls(
        fn_swapped,
        maxiter=tmax,
        tol=eta,
        has_aux=False,
        implicit_diff=False,
        unroll=unroll,
        jit=True,
        verbose=verbose,
        **solver_kwargs,
    )

    result = solver.run(x_init, a)

    return result.params
