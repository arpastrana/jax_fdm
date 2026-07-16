"""
Routing tests for the equilibrium backward pass.

These lock *which* differentiation path a model configuration executes, rather
than the numbers it produces. The gradient-correctness tests
(test_taylor_convergence.py) would pass identically no matter which internal
solver ran, so a refactor could silently reroute the backward pass and stay
green. Here we spy on the routing functions and assert the expected one fires:

- tmax == 1                     -> single linear solve, the iterative fixed-point
                                   solver is never entered.
- tmax > 1, implicit_diff=True  -> iterative solve differentiated by the implicit
                                   adjoint, which runs the lineax linear_solve in
                                   fixed_point_bwd_adjoint (the Normal(CG) path).
- tmax > 1, implicit_diff=False -> iterative solve differentiated by unrolled
                                   autodiff; the adjoint linear_solve is not hit.

The adjoint path is checked for both dense and sparse stiffness matrices, which
take different branches inside fixed_point_bwd_adjoint but converge on the same
lineax solve. Assertions are on execution only (call/not-called), so the tests
are independent of backend, BLAS, and precision.
"""

import contextlib
import io
from unittest import mock

import jax.numpy as jnp

import jax_fdm.equilibrium.models as models_module
import jax_fdm.equilibrium.solvers.fixed_point as fixed_point_module
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import model_from_sparsity
from jax_fdm.equilibrium import structure_from_datastructure
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter


def _value_and_grad_fn(tmax, implicit_diff, sparse):
    """
    Build jit-free (loss, grad) over the flat parameters and the initial vector.

    A meshgrid under follower face loads (is_load_local=True) makes equilibrium a
    shape-dependent fixed point, so tmax > 1 exercises the iterative solver whose
    adjoint runs the lineax linear solve. jit_fn=False keeps the routing functions
    interceptable: a jitted problem would trace and cache before the spies patch in.
    """
    mesh = FDMesh.from_meshgrid(dx=2, nx=5)
    mesh.vertices_supports([vertex for vertex in mesh.vertices_on_boundary()])
    mesh.edges_forcedensities(-2.0, keys=mesh.edges())
    mesh.faces_loads([0.0, 0.0, -0.1], keys=mesh.faces())

    parameters = [EdgeForceDensityParameter(edge, -50.0, -0.01)
                  for edge in mesh.edges()]

    model = model_from_sparsity(sparse=sparse,
                                tmax=tmax,
                                eta=1e-6,
                                is_load_local=True,
                                implicit_diff=implicit_diff)
    structure = structure_from_datastructure(mesh, sparse=sparse)

    goals = [EdgeLengthGoal(edge, target=1.0) for edge in mesh.edges()]
    loss = Loss(SquaredError(goals=goals))

    with contextlib.redirect_stdout(io.StringIO()):
        problem = LBFGSB().problem(model,
                                   structure,
                                   mesh,
                                   loss,
                                   parameters,
                                   maxiter=1,
                                   tol=1e-6,
                                   jit_fn=False)

    return problem.fun, jnp.asarray(problem.x0)


def _routing(tmax, implicit_diff, sparse):
    """
    Return (entered_iterative_solver, ran_adjoint_linear_solve) for a config.

    The forward spy wraps solver_fixedpoint_implicit, called only when the model
    routes through the implicit iterative solver. The adjoint spy wraps the lineax
    linear_solve looked up as a module global inside fixed_point_bwd_adjoint, so it
    fires only when the implicit adjoint backward pass runs. Both must be patched
    before the problem builds, since the backward rule traces on the first call.
    """
    with mock.patch.object(models_module,
                           "solver_fixedpoint_implicit",
                           wraps=models_module.solver_fixedpoint_implicit) as spy_forward, \
         mock.patch.object(fixed_point_module,
                           "linear_solve",
                           wraps=fixed_point_module.linear_solve) as spy_adjoint:
        fun, x0 = _value_and_grad_fn(tmax, implicit_diff, sparse)
        fun(x0)

        return spy_forward.called, spy_adjoint.called


def test_linear_path_skips_iterative_solver():
    """
    tmax == 1 solves in a single pass; the iterative solver is never entered.
    """
    entered_iterative, ran_adjoint = _routing(tmax=1, implicit_diff=True, sparse=False)

    assert not entered_iterative
    assert not ran_adjoint


def test_implicit_path_runs_adjoint_linear_solve():
    """
    tmax > 1 with implicit_diff routes through the implicit adjoint linear solve.
    """
    entered_iterative, ran_adjoint = _routing(tmax=50, implicit_diff=True, sparse=False)

    assert entered_iterative
    assert ran_adjoint


def test_unrolled_path_skips_adjoint_linear_solve():
    """
    tmax > 1 without implicit_diff unrolls autodiff, bypassing the adjoint solve.
    """
    entered_iterative, ran_adjoint = _routing(tmax=50, implicit_diff=False, sparse=False)

    assert not entered_iterative
    assert not ran_adjoint


def test_sparse_implicit_path_runs_adjoint_linear_solve():
    """
    The sparse stiffness branch of the adjoint still runs the same lineax solve.
    """
    entered_iterative, ran_adjoint = _routing(tmax=50, implicit_diff=True, sparse=True)

    assert entered_iterative
    assert ran_adjoint
