try:
    from optimistix import LevenbergMarquardt
    from optimistix import Dogleg
    from optimistix import least_squares
    from optimistix import ImplicitAdjoint
    from optimistix import RecursiveCheckpointAdjoint

    import optimistix as optx
    import lineax as lx

except ImportError:
    pass


def solver_levenberg_marquardt_optimistix(fn, theta, x_init, solver_config):
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

    if implicit_diff:
        adjoint = ImplicitAdjoint()
    else:
        adjoint = RecursiveCheckpointAdjoint()

    def fn_swapped(x, theta):
        return fn(theta, x)

    stats = {}
    if verbose:
        stats = {"loss", "step_size"}

    verbose = frozenset(stats)

    # solver = LevenbergMarquardt(
    solver = Dogleg(
        rtol=eta,
        atol=eta,
        norm=optx.two_norm,
        verbose=verbose,
        # linear_solver=lx.NormalCG(eta, eta)
    )

    solution = least_squares(
        fn=fn_swapped,
        solver=solver,
        y0=x_init,
        args=theta,
        has_aux=False,
        max_steps=tmax,
        throw=False,
        # tags=frozenset({lx.positive_semidefinite_tag})
    )

    return solution.value
