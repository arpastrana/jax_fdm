try:
    from optimistix import ImplicitAdjoint
    from optimistix import RecursiveCheckpointAdjoint

except ImportError:
    pass


def solver_optimistix(solver_cls, routine_fn, fn, a, x_init, solver_config, solver_kwargs=None):
    """
    Find a root of a function f(a, x) with optimistix.

    Parameters
    ----------
    fn : The function to iterate upon.
    a : The function parameters.
    x_init: An initial guess for the values of the solution vector.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x : The solution vector at a fixed point.

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
        adjoint=adjoint
    )

    return solution.value
