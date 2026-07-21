"""Shared type aliases for the iterative equilibrium solvers."""

from typing import Any

__all__ = ["SolverIterParams"]


# The parameter PyTree threaded through the fixed-point, nonlinear, and
# least-squares solvers: e.g. (K, R_fixed, xyz_fixed, load_state) on the
# fixed-point path or (q, xyz_fixed, load_state) on the residual path. Its
# leaves are heterogeneous (a dense Array or sparse CSC stiffness matrix, plain
# arrays, and a LoadState) and its arity varies by path, so the honest shared
# type is a variadic tuple. Distinct from the parameters fed to the model call.
SolverIterParams = tuple[Any, ...]
