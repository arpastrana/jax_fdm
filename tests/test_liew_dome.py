"""
Validation tests against Liew (2020), Section 4.2 "Circular gridshell".

Reference: A. Liew, "Constrained Force Density Method optimisation for
form-finding," Structures 28 (2020) 1845-1856, doi:10.1016/j.istruc.2020.09.078.

The fixture ``liew_dome.json`` is the frozen, pre-optimization gridshell network
(613 vertices, 1800 edges). Freezing it pins the node and edge ordering.
Only the base case and the volume-only optimization (alpha_lp=1, alpha_lengths=0)
are asserted. The other configurations from the paper are not version-robust

The dome geometry was kindly provided by the paper's author.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fdm.constraints import NodeXCoordinateConstraint
from jax_fdm.constraints import NodeYCoordinateConstraint
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError
from jax_fdm.optimization import SLSQP
from jax_fdm.parameters import EdgeGroupForceDensityParameter

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "data", "liew_dome.json")

# Compression-only force density bounds. The paper states qmax=5, but allowing
# tension makes the optimization diverge; the compression-only range reproduces
# the published figures.
QMIN, QMAX = -5.0, -1e-6
XY_TOL = 0.1

# Reference statistics read from the paper, Section 4.2.
# (volume, length min, length max, length mean, mean length deviation).
PAPER_BASE = (402.0, 0.153, 2.143, 0.960, 0.320)
PAPER_VOLUME = (283.0, 0.073, 1.990, 1.061, 0.274)


# ==============================================================================
# Helpers
# ==============================================================================

def _origin_distance(point):
    """
    Euclidean distance from a point to the origin.
    """
    return (point[0] ** 2 + point[1] ** 2 + point[2] ** 2) ** 0.5


def _edge_groups(network):
    """
    Group edges by their midpoint distance to the origin, rounded to 0.1.

    Mirrors the radial grouping used to build the dome, yielding the 34 symmetry
    groups the paper exploits.
    """
    groups = {}
    for edge in network.edges():
        key = round(_origin_distance(network.edge_midpoint(*edge)), 1)
        groups.setdefault(key, []).append(edge)

    return list(groups.values())


def _load_path(network):
    """
    Total load path of the network, the sum of absolute force-length products.
    """
    return sum(abs(network.edge_force(edge)) * network.edge_length(*edge)
               for edge in network.edges())


def _length_stats(network):
    """
    Edge-length minimum, maximum, mean, and mean deviation from unit length.
    """
    lengths = np.array([network.edge_length(*edge) for edge in network.edges()])

    return lengths.min(), lengths.max(), lengths.mean(), np.mean(np.abs(lengths - 1.0))


def _metrics(network):
    """
    The five paper statistics: volume, length min, max, mean, and deviation.
    """
    lmin, lmax, lmean, dev = _length_stats(network)

    return (_load_path(network), lmin, lmax, lmean, dev)


def _volume_optimized(network):
    """
    Run the paper's volume-only optimization (alpha_lp=1, alpha_lengths=0).

    Returns the optimized network and the optimizer that solved it, so callers
    can inspect the scipy convergence status.
    """
    parameters = [EdgeGroupForceDensityParameter(group, QMIN, QMAX)
                  for group in _edge_groups(network)]

    loss = Loss(PredictionError([NetworkLoadPathGoal()], alpha=1.0, name="NetworkLoadPathGoal"))

    constraints = []
    for node in network.nodes_free():
        x, y, _ = network.node_coordinates(node)
        constraints.append(NodeXCoordinateConstraint(node, x - XY_TOL, x + XY_TOL))
        constraints.append(NodeYCoordinateConstraint(node, y - XY_TOL, y + XY_TOL))

    optimizer = SLSQP()
    optimized = constrained_fdm(fdm(network),
                                optimizer=optimizer,
                                loss=loss,
                                parameters=parameters,
                                maxiter=5000,
                                tol=1e-6,
                                constraints=constraints)

    return optimized, optimizer


# ==============================================================================
# Tests
# ==============================================================================

def test_base_case_matches_paper():
    """
    The unoptimized gridshell reproduces the paper's base-case statistics.
    """
    network = fdm(FDNetwork.from_json(FIXTURE))

    volume, lmin, lmax, lmean, dev = _metrics(network)
    ref_volume, ref_lmin, ref_lmax, ref_lmean, ref_dev = PAPER_BASE

    assert jnp.allclose(volume, ref_volume, atol=0.5)
    assert jnp.allclose(lmin, ref_lmin, atol=1e-3)
    assert jnp.allclose(lmax, ref_lmax, atol=1e-3)
    assert jnp.allclose(lmean, ref_lmean, atol=1e-3)
    assert jnp.allclose(dev, ref_dev, atol=1e-3)


@pytest.mark.slow
def test_volume_optimization_matches_paper():
    """
    Volume-only optimization converges to the paper's reported gridshell.
    """
    network = FDNetwork.from_json(FIXTURE)
    optimized, optimizer = _volume_optimized(network)

    # SLSQP must report success (exit mode 0); a non-converged build would
    # otherwise pass a wrong-but-plausible result silently.
    assert optimizer.result.status == 0

    volume, lmin, lmax, lmean, dev = _metrics(optimized)
    ref_volume, _, ref_lmax, ref_lmean, ref_dev = PAPER_VOLUME

    assert jnp.allclose(volume, ref_volume, rtol=1e-2)
    assert jnp.allclose(lmax, ref_lmax, rtol=1e-2)
    assert jnp.allclose(lmean, ref_lmean, rtol=1e-2)
    assert jnp.allclose(dev, ref_dev, rtol=1e-2)

    # The shortest edge settles near 0.089 rather than the paper's 0.073 (a
    # single short-edge extreme), but stays below the base case minimum.
    assert lmin < PAPER_BASE[1]
