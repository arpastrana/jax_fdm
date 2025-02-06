from jax.tree_util import Partial


def solver_jaxopt(solver_cls, f, solver_config, solver_kwargs=None):
    """
    Solve for a fixed point x* of a function f(x, theta) using a jaxopt solver.

    Parameters
    ----------
    f : The function to iterate upon.
    x_init: An initial guess for the values of the solution vector.
    theta : The function parameters.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at a fixed point.
    """
    tmax = solver_config["tmax"]
    eta = solver_config["eta"]
    verbose = solver_config["verbose"]
    implicit_diff = solver_config["implicit_diff"]

    if solver_kwargs is None:
        solver_kwargs = {}

    # NOTE: Unroll python loop if solver config disables implicit diff
    # This enables reverse-mode AD to calculate gradients when implicit differentiation
    # is off because the solver uses lax.while_loop under the hood to not unroll iterations
    # but this type of while loop is not reverse-mode differentiable.
    unroll = False
    if not implicit_diff:
        unroll = True

    solver = solver_cls(
        f,
        maxiter=tmax,
        tol=eta,
        has_aux=False,
        implicit_diff=False,  # False, NOTE: Disabling jaxopt implicit diff on purpose
        unroll=unroll,   # unroll
        jit=True,
        verbose=verbose,
        **solver_kwargs
    )

    return Partial(solver_jaxopt_run, solver=solver)


def solver_jaxopt_run(solver, x_init, theta, structure):
    """
    Run a jaxopt solver and extract the found solution.
    """
    result = solver.run(x_init, theta, structure)

    return result.params
