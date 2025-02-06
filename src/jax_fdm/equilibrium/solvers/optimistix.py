from jax.tree_util import Partial

try:
    from optimistix import Newton
    from optimistix import NonlinearCG
    from optimistix import BFGS
    from optimistix import Dogleg
    from optimistix import LevenbergMarquardt
    from optimistix import ImplicitAdjoint
    from optimistix import RecursiveCheckpointAdjoint
    from optimistix import two_norm
    from optimistix import least_squares
    from optimistix import minimise
    from optimistix import root_find

    import lineax as lx

except ImportError:
    pass


def is_solver_minimization(solver):
    """
    Test if a solver function is a minimization solver.

    Parameters
    ----------
    `solver_fn`: A solver function

    Returns
    -------
    `True` if the solver is a least squares solver. Otherwise, `False`.
    """
    solvers = {
        solver_bfgs_optimistix,
        solver_nonlinear_cg_optimistix,
    }

    return solver in solvers


def is_solver_root_finding(solver):
    """
    Test if a solver function is a root finding solver.

    Parameters
    ----------
    `solver_fn`: A solver function

    Returns
    -------
    `True` if the solver is a least squares solver. Otherwise, `False`.
    """
    solvers = {
        solver_newton_optimistix,
    }

    return solver in solvers


def solver_levenberg_marquardt_optimistix(fn, solver_config):
    """
    """
    eta = solver_config["eta"]
    solver_kwargs = {"linear_solver": lx.NormalCG(eta, eta)}
    return solver_optimistix(LevenbergMarquardt, lsq_fn, fn, solver_config, solver_kwargs)


def solver_dogleg_optimistix(fn, solver_config):
    """
    """
    eta = solver_config["eta"]
    solver_kwargs = {"linear_solver": lx.NormalCG(eta, eta)}
    return solver_optimistix(Dogleg, lsq_fn, fn, solver_config, solver_kwargs)


def solver_bfgs_optimistix(fn, solver_config):
    """
    """
    solver_kwargs = {"use_inverse": True}
    return solver_optimistix(BFGS, min_fn, fn, solver_config, solver_kwargs)


def solver_nonlinear_cg_optimistix(fn, solver_config):
    """
    """
    solver_kwargs = {}
    return solver_optimistix(NonlinearCG, min_fn, fn, solver_config, solver_kwargs)


def solver_newton_optimistix(fn, solver_config):
    """
    """
    eta = solver_config["eta"]
    solver_kwargs = {}
    return solver_optimistix(Newton, root_fn, fn, solver_config, solver_kwargs)


def solver_optimistix(solver_cls, routine_fn, fn, solver_config, solver_kwargs=None):
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

    verbose_free = {NonlinearCG, Newton}
    if verbose and solver_cls not in verbose_free:
        stats = {"loss", "step_size"}
        verbose = frozenset(stats)
        solver_kwargs["verbose"] = verbose

    solver = solver_cls(
        rtol=eta,
        atol=eta,
        **solver_kwargs,
    )

    return Partial(routine_fn, solver=solver, tmax=tmax, adjoint=adjoint, fn=fn)


def lsq_fn(solver, fn, tmax, adjoint, x_init, theta, structure):
    """
    """
    # NOTE: Creating a lambda or a function closure triggers recompilation!
    # But using jtu.Partial fixes that issue.
    _fn = Partial(fn, structure=structure)

    solution = least_squares(
        fn=_fn,
        solver=solver,
        y0=x_init,
        args=theta,
        has_aux=False,
        max_steps=tmax,
        throw=False,
        adjoint=adjoint
    )

    return solution.value


def min_fn(solver, fn, tmax, adjoint, x_init, theta, structure):
    """
    """
    # NOTE: Creating a lambda or a function closure triggers recompilation!
    # But using jtu.Partial fixes that issue.
    _fn = Partial(fn, structure=structure)

    solution = minimise(
        fn=_fn,
        solver=solver,
        y0=x_init,
        args=theta,
        has_aux=False,
        max_steps=tmax,
        throw=False,
        adjoint=adjoint
    )

    return solution.value


def root_fn(solver, fn, tmax, adjoint, x_init, theta, structure):
    """
    """
    # NOTE: Creating a lambda or a function closure triggers recompilation!
    # But using jtu.Partial fixes that issue.
    _fn = Partial(fn, structure=structure)

    solution = root_find(
        fn=_fn,
        solver=solver,
        y0=x_init,
        args=theta,
        has_aux=False,
        max_steps=tmax,
        throw=False,
        adjoint=adjoint
    )

    return solution.value
