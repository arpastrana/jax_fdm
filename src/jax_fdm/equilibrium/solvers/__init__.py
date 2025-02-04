from .jaxopt import *  # noqa F403
# from .optimistix import *  # noqa F403
from .fixed_point import *  # noqa F403
from .least_squares import *  # noqa F403


SOLVERS = {
    "fixed_point": solver_fixedpoint,
    "anderson": solver_anderson,
    "gauss_newton": solver_gauss_newton,
    "levenberg_marquardt": solver_levenberg_marquardt,
    "lbfgs": solver_lbfgs,
    "lbfgs_scipy": solver_lbfgs_scipy,
}

__all__ = [name for name in dir() if not name.startswith('_')]
