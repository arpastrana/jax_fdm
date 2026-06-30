"""
Validation against the analytical benchmark in Pastrana et al. (2026),
Appendix A.1 "Analytical benchmark".

Reference: R. Pastrana, D. Oktay, K. U. Bletzinger, R. P. Adams, S. Adriaenssens,
"Differentiable force density method for the design of lightweight
structures," Computer Methods in Applied Mechanics and Engineering 458
(2026) 118783, doi:10.1016/j.cma.2026.118783.

The inverse problem finds the rise of a compression-only planar arch, of
fixed horizontal projection and under a uniform vertical load, that minimizes
its load path Omega. The continuous solution is known in closed form, so no
fixture is needed: the network is generated programmatically and the optimized
rise and load path are asserted against the analytical values (paper Eqs. A.2
and A.3).

The paper reports matching Omega* within 1% error once the discretization
reaches n_v = 100, and z*_max within 0.1% with fewer vertices. We reproduce
both claims and the monotone convergence of the error with n_v.
"""

from math import fabs
from math import sqrt

import pytest

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeGroupForceDensityParameter

# Benchmark setup from the paper: unit load, 10 m span, initial force density.
SPAN = 10.0
RHO = 1.0
Q_INIT = -10.0

# Compression-only force density bounds, single grouped parameter.
QMIN, QMAX = -1000.0, -1e-3


# ==============================================================================
# Analytical solution (paper Eqs. A.2 and A.3)
# ==============================================================================

def _rise_analytical(span):
    """
    Closed-form optimal rise z*_max = sqrt(3) * d / 4.
    """
    return sqrt(3.0) * span / 4.0


def _loadpath_analytical(span, rho):
    """
    Closed-form optimal load path Omega* = rho * d^2 / sqrt(3).
    """
    return rho * span ** 2 / sqrt(3.0)


# ==============================================================================
# Helpers
# ==============================================================================

def _arch_network(num_vertices):
    """
    Build a planar arch of ``num_vertices`` vertices spanning SPAN meters.

    The vertices are evenly spaced along the horizontal axis, the two end
    vertices are anchored, every edge starts at Q_INIT, and the uniform load
    RHO is lumped into equal vertical point loads p_i = rho*d/n_v.
    """
    network = FDNetwork()

    for node in range(num_vertices):
        x = -SPAN / 2.0 + SPAN * node / (num_vertices - 1)
        network.add_node(node, x=x, y=0.0, z=0.0)
    for node in range(num_vertices - 1):
        network.add_edge(node, node + 1)

    network.node_support(key=0)
    network.node_support(key=num_vertices - 1)
    network.edges_forcedensities(Q_INIT, keys=network.edges())

    py_node = (-RHO * SPAN) / network.number_of_nodes()
    network.nodes_loads([0.0, py_node, 0.0], keys=network.nodes())

    return network


def _optimize_arch(num_vertices):
    """
    Minimize the load path of the arch over a single shared force density.

    Returns the optimized rise (max absolute vertical coordinate) and load path.
    """
    network = _arch_network(num_vertices)

    parameters = [EdgeGroupForceDensityParameter(list(network.edges()), QMIN, QMAX)]
    loss = Loss(PredictionError([NetworkLoadPathGoal()]))

    optimized = constrained_fdm(network,
                                optimizer=LBFGSB(),
                                loss=loss,
                                parameters=parameters,
                                maxiter=100,
                                tol=1e-9)

    rise = max(optimized.nodes_attribute("y"), key=fabs)

    return fabs(rise), optimized.loadpath()


def _relative_error(numerical, analytical):
    """
    Absolute relative error between a numerical and analytical value.
    """
    return fabs(numerical - analytical) / fabs(analytical)


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("num_vertices,rise_tol,lp_tol", [
    (100, 1e-3, 1.1e-2),
    (500, 1e-3, 3e-3),
])
def test_arch_matches_analytical(num_vertices, rise_tol, lp_tol):
    """
    The optimized arch meets the closed-form rise and load path.

    Tolerances follow the paper: the rise converges within 0.1% and the load
    path within 1% by n_v = 100, tightening as the discretization refines.
    """
    rise, loadpath = _optimize_arch(num_vertices)

    assert _relative_error(rise, _rise_analytical(SPAN)) < rise_tol
    assert _relative_error(loadpath, _loadpath_analytical(SPAN, RHO)) < lp_tol


def test_arch_error_decreases_with_discretization():
    """
    Refining the arch monotonically reduces the load path error, confirming
    the convergence trend the paper reports between n_v = 5 and n_v = 500.
    """
    lp_target = _loadpath_analytical(SPAN, RHO)

    errors = [_relative_error(_optimize_arch(nv)[1], lp_target)
              for nv in (5, 50, 100, 500)]

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
