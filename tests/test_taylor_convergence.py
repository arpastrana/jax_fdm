"""
Validation against the adjoint-correctness test in Pastrana et al. (2026),
Appendix A.4 "Correctness of the analytical adjoints".

Reference: R. Pastrana, D. Oktay, K. U. Bletzinger, R. P. Adams, S. Adriaenssens,
"Differentiable force density method for the design of lightweight
structures," Computer Methods in Applied Mechanics and Engineering 458
(2026) 118783, doi:10.1016/j.cma.2026.118783.

The Taylor remainder convergence test verifies a gradient formulation: for a
correct gradient, the second-order remainder

    rho(mu) = | L(theta + mu*v) - L(theta) - mu * v^T grad L(theta) |

decays as O(mu^2), i.e. with a log-log slope of 2 as the step mu is halved,
until round-off dominates. The paper applies it to confirm the analytical
adjoints (ADA): the linear-solver adjoint (a single equilibrium solve, tmax=1)
and the unrolled recursive solver (URS) adjoint (an iterative fixed-point solve
under shape-dependent follower loads, tmax>1, is_load_local=True).

The analytical adjoints are a property of the equilibrium model, not of any
specific structure, so we exercise the same code paths on compact, generated
geometry instead of the paper's heavy domes -- keeping the suite COMPAS-free and
fixture-free. To obtain a scalar loss and its gradient over the flat parameter
vector without running an optimization, we reuse the optimizer's own problem
hook (``Optimizer.problem(...)["fun"]`` is ``jit(value_and_grad(loss))``), which
is exactly how the paper's experiment scripts run their Taylor tests. Each test
asserts both the paper's claim (slope -> 2) and that the analytical-adjoint
gradient matches an unrolled automatic-differentiation gradient (implicit_diff
True vs False).
"""

import contextlib
import io

import jax
import jax.numpy as jnp
import pytest

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import model_from_sparsity
from jax_fdm.equilibrium import structure_from_datastructure
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter

# A fixed key keeps the random perturbation directions deterministic.
KEY = jax.random.PRNGKey(42)

# Taylor sweep: ten unit directions, halving the step from 1e-2, as in the paper.
NUM_DIRS = 10
NUM_EPS = 8
EPS0 = 1e-2

# The last few remainders flatten out once round-off dominates (rho < ~1e-11),
# so the slope is measured over the well-behaved head of the sweep.
TAIL_DROP = 3

# Slope tolerance. The probe gave 2.000 in every configuration; 0.1 leaves margin
# for the mild deviation the paper itself reports near round-off.
SLOPE_TOL = 0.1


# ==============================================================================
# Taylor remainder helpers (ported from the paper's experiment scripts)
# ==============================================================================

def _taylor_remainder(fun, x0, f0, g0, v, eps):
    """
    Second-order Taylor remainder |f(x0 + eps*v) - f0 - eps * g0^T v|.
    """
    f_eps, _ = fun(x0 + eps * v)

    return jnp.abs(f_eps - f0 - eps * jnp.vdot(g0, v))


def _convergence_slope(remainders):
    """
    Log-log slope between successive remainders of a halved step size.

    With each step half the previous one, the order is log2 of the remainder
    ratio: a second-order remainder quarters as the step halves, giving 2.
    """
    return jnp.log2(remainders[:-1] / remainders[1:])


def _mean_taylor_slope(fun, x0, key):
    """
    Mean second-order convergence slope over NUM_DIRS random unit directions.

    The remainder is measured at NUM_EPS step sizes halved from EPS0, and the
    slope is averaged over the pre-round-off head of each sweep.
    """
    f0, g0 = fun(x0)
    eps = jnp.asarray([EPS0 * (0.5 ** i) for i in range(NUM_EPS)])

    slopes = []
    for _ in range(NUM_DIRS):
        key, subkey = jax.random.split(key)
        v = jax.random.uniform(subkey, x0.shape)
        v = v / jnp.linalg.norm(v)

        remainders = jnp.asarray([_taylor_remainder(fun, x0, f0, g0, v, e) for e in eps])
        slope = _convergence_slope(remainders[:-TAIL_DROP])
        slopes.append(jnp.mean(slope))

    return float(jnp.mean(jnp.asarray(slopes)))


# ==============================================================================
# Problem builders
# ==============================================================================

def _loss_and_grad(datastructure, parameters, tmax, is_load_local, implicit_diff):
    """
    Build the (loss, gradient) function over the flat parameter vector and the
    initial parameters, selecting the adjoint via the equilibrium model.

    Reuses the optimizer's problem hook, whose "fun" is jit(value_and_grad(loss)).
    The model flags pick the adjoint under test: tmax=1 is the linear-solver
    adjoint, tmax>1 with follower loads is the URS adjoint, and implicit_diff
    toggles the analytical adjoint (True) against unrolled autodiff (False).
    """
    model = model_from_sparsity(sparse=False,
                                tmax=tmax,
                                eta=1e-6,
                                is_load_local=is_load_local,
                                implicit_diff=implicit_diff)
    structure = structure_from_datastructure(datastructure, sparse=False)

    goals = [EdgeLengthGoal(edge, target=1.0) for edge in datastructure.edges()]
    loss = Loss(SquaredError(goals=goals))

    # Suppress the optimizer's warm-up prints to keep the test output clean.
    with contextlib.redirect_stdout(io.StringIO()):
        problem = LBFGSB().problem(model,
                                   structure,
                                   datastructure,
                                   loss,
                                   parameters,
                                   maxiter=1,
                                   tol=1e-6,
                                   jit_fn=True)

    return problem["fun"], jnp.asarray(problem["x0"])


def _linear_problem(num_vertices, implicit_diff):
    """
    A planar arch solved in a single pass (tmax=1, linear-solver adjoint).

    Force densities over every edge are the parameters; the loss drives edge
    lengths toward unity. Built from scratch to stay COMPAS-free.
    """
    network = FDNetwork()
    for node in range(num_vertices):
        x = -5.0 + 10.0 * node / (num_vertices - 1)
        network.add_node(node, x=x, y=0.0, z=0.0)
    for node in range(num_vertices - 1):
        network.add_edge(node, node + 1)

    network.node_support(key=0)
    network.node_support(key=num_vertices - 1)
    network.edges_forcedensities(-2.0, keys=network.edges())
    network.nodes_loads([0.0, -0.1, 0.0], keys=network.nodes())

    parameters = [EdgeForceDensityParameter(edge, -50.0, -0.01)
                  for edge in network.edges()]

    return _loss_and_grad(network, parameters,
                          tmax=1, is_load_local=False, implicit_diff=implicit_diff)


def _urs_problem(nx, implicit_diff):
    """
    A meshgrid under follower face loads, form-found iteratively (tmax=50,
    is_load_local=True, URS adjoint).

    The shape-dependent loads make the equilibrium a fixed point the URS adjoint
    must backpropagate through. Boundary vertices are supported.
    """
    mesh = FDMesh.from_meshgrid(dx=2, nx=nx)
    mesh.vertices_supports([vertex for vertex in mesh.vertices_on_boundary()])
    mesh.edges_forcedensities(-2.0, keys=mesh.edges())
    mesh.faces_loads([0.0, 0.0, -0.1], keys=mesh.faces())

    parameters = [EdgeForceDensityParameter(edge, -50.0, -0.01)
                  for edge in mesh.edges()]

    return _loss_and_grad(mesh, parameters,
                          tmax=50, is_load_local=True, implicit_diff=implicit_diff)


# ==============================================================================
# Tests: linear-solver adjoint (tmax=1)
# ==============================================================================

@pytest.mark.parametrize("num_vertices", [
    101,
    pytest.param(501, marks=pytest.mark.slow),
])
def test_linear_adjoint_second_order_convergence(num_vertices):
    """
    The linear-solver adjoint gradient passes the Taylor test (slope -> 2).
    """
    fun, x0 = _linear_problem(num_vertices, implicit_diff=True)

    slope = _mean_taylor_slope(fun, x0, KEY)

    assert slope == pytest.approx(2.0, abs=SLOPE_TOL)


@pytest.mark.parametrize("num_vertices", [
    101,
    pytest.param(501, marks=pytest.mark.slow),
])
def test_linear_adjoint_matches_autodiff(num_vertices):
    """
    The analytical linear-solver adjoint gradient matches unrolled autodiff.
    """
    fun_ada, x0 = _linear_problem(num_vertices, implicit_diff=True)
    fun_ad, _ = _linear_problem(num_vertices, implicit_diff=False)

    _, grad_ada = fun_ada(x0)
    _, grad_ad = fun_ad(x0)

    assert jnp.allclose(grad_ada, grad_ad, atol=1e-8)


# ==============================================================================
# Tests: URS adjoint with follower loads (tmax>1, is_load_local=True)
# ==============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("nx", [
    5,
    pytest.param(21, marks=pytest.mark.slow),
])
def test_urs_adjoint_second_order_convergence(nx):
    """
    The URS adjoint gradient under follower loads passes the Taylor test.

    The nx=21 tier exercises a system with 1,200 free degrees of freedom.
    """
    fun, x0 = _urs_problem(nx, implicit_diff=True)

    slope = _mean_taylor_slope(fun, x0, KEY)

    assert slope == pytest.approx(2.0, abs=SLOPE_TOL)


@pytest.mark.slow
@pytest.mark.parametrize("nx", [
    5,
    pytest.param(21, marks=pytest.mark.slow),
])
def test_urs_adjoint_matches_autodiff(nx):
    """
    The analytical URS adjoint gradient matches unrolled autodiff.
    """
    fun_ada, x0 = _urs_problem(nx, implicit_diff=True)
    fun_ad, _ = _urs_problem(nx, implicit_diff=False)

    _, grad_ada = fun_ada(x0)
    _, grad_ad = fun_ad(x0)

    assert jnp.allclose(grad_ada, grad_ad, atol=1e-7)


# ==============================================================================
# Negative control: a wrong gradient must fail the Taylor test
# ==============================================================================

def test_corrupted_gradient_fails_convergence():
    """
    A gradient nudged by a tiny constant no longer passes the Taylor test.

    Adding 1e-3 to the correct gradient leaves a linear error term in the
    remainder, so the second-order decay collapses (the slope falls toward 1).
    This confirms the convergence assertion actually detects a bad gradient
    rather than passing trivially.
    """
    fun, x0 = _linear_problem(101, implicit_diff=True)

    def corrupted(x):
        loss, grad = fun(x)
        return loss, grad + 1e-3

    slope = _mean_taylor_slope(corrupted, x0, KEY)

    assert slope != pytest.approx(2.0, abs=SLOPE_TOL)
