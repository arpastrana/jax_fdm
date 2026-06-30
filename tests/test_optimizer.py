"""
Characterization tests for constrained form-finding optimization.
"""

import jax.numpy as jnp
from conftest import assert_baseline

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import EdgeForceDensityParameter

TARGET_LENGTH = 0.6
BOUND_LOW = -10.0
BOUND_UP = -0.1
MAXITER = 50


def _optimize(network):
    """
    Drive every edge length toward a common target by tuning force densities.
    """
    goals = [EdgeLengthGoal(edge, target=TARGET_LENGTH) for edge in network.edges()]
    loss = Loss(SquaredError(goals=goals))
    parameters = [EdgeForceDensityParameter(edge, BOUND_LOW, BOUND_UP)
                  for edge in network.edges()]

    return constrained_fdm(network,
                           optimizer=SLSQP(),
                           loss=loss,
                           parameters=parameters,
                           maxiter=MAXITER)


def test_optimizer_reduces_length_error(arch_network):
    """
    Optimization lowers the edge-length error relative to the initial solve.
    """
    lengths_before = jnp.array(fdm(arch_network).edges_lengths())
    lengths_after = jnp.array(_optimize(arch_network).edges_lengths())

    error_before = jnp.sum((lengths_before - TARGET_LENGTH) ** 2)
    error_after = jnp.sum((lengths_after - TARGET_LENGTH) ** 2)

    assert error_after < error_before


def test_optimizer_respects_bounds(arch_network):
    """
    The optimized force densities stay within their bounds.
    """
    optimized = _optimize(arch_network)

    forcedensities = jnp.array([optimized.edge_forcedensity(edge)
                                for edge in optimized.edges()])

    assert jnp.all(forcedensities >= BOUND_LOW)
    assert jnp.all(forcedensities <= BOUND_UP)


def test_optimizer_result_in_equilibrium(arch_network):
    """
    The optimized network is still in static equilibrium at its free nodes.
    """
    optimized = _optimize(arch_network)

    residuals = jnp.array(optimized.nodes_residual(keys=optimized.nodes_free()))

    assert jnp.allclose(residuals, 0.0, atol=1e-9)


def test_optimizer_forcedensities_baseline(arch_network):
    """
    The converged force densities reproduce the captured baseline.
    """
    optimized = _optimize(arch_network)

    forcedensities = [optimized.edge_forcedensity(edge)
                      for edge in optimized.edges()]

    assert_baseline("arch_forcedensities", forcedensities)
