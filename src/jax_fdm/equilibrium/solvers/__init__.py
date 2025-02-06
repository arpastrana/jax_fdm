from .jaxopt import *  # noqa F403
from .optimistix import *  # noqa F403
from .fixed_point import *  # noqa F403
from .least_squares import *  # noqa F403


SOLVERS = {
    "fixed_point": solver_fixedpoint,
    "anderson": solver_anderson,
    "gauss_newton": solver_gauss_newton,
    "levenberg_marquardt": solver_levenberg_marquardt,
    "lbfgs": solver_lbfgs,
    "lbfgs_scipy": solver_lbfgs_scipy,
    "levenberg_marquardt_optimistix": solver_levenberg_marquardt_optimistix,
    "dogleg_optimistix": solver_levenberg_marquardt_optimistix,
    "bfgs_optimistix": solver_bfgs_optimistix,
    "nonlinear_cg_optimistix": solver_nonlinear_cg_optimistix,
    "newton_optimistix": solver_newton_optimistix,
}

__all__ = [name for name in dir() if not name.startswith('_')]
