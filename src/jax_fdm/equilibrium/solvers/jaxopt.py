def solver_jaxopt(solver_cls, fn, a, x_init, solver_config, solver_kwargs=None):
    """
    Solve for a fixed point of a function f(a, x) using anderson acceleration in jaxopt.

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

    if solver_kwargs is None:
        solver_kwargs = {}

    # NOTE: Unroll python loop if solver config disables implicit diff
    # This enables reverse-mode AD to calculate gradients when implicit differentiation
    # is off because the solver uses lax.while_loop under the hood to not unroll iterations
    # but this type of while loop is not reverse-mode differentiable.
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
        **solver_kwargs
    )

    result = solver.run(x_init, a)

    return result.params
