"""
Characterization tests for the force density solver.
"""

import jax.numpy as jnp

from conftest import assert_baseline
from jax_fdm.equilibrium import fdm


def test_arch_free_nodes_in_equilibrium(arch_network):
    """
    The residual force vanishes at every free node of the solved arch.
    """
    eq_network = fdm(arch_network)

    residuals = jnp.array(eq_network.nodes_residual(keys=eq_network.nodes_free()))

    assert jnp.allclose(residuals, 0.0, atol=1e-9)


def test_arch_reactions_balance_loads(arch_network):
    """
    The support reactions balance the total applied load.
    """
    eq_network = fdm(arch_network)

    loads = jnp.array(eq_network.nodes_loads(keys=eq_network.nodes_free()))
    reactions = jnp.array(eq_network.nodes_reactions())

    assert jnp.allclose(
        jnp.abs(jnp.sum(reactions, axis=0)),
        jnp.abs(jnp.sum(loads, axis=0)),
        atol=1e-9,
    )


def test_arch_profile_symmetric(arch_network):
    """
    The solved arch is symmetric about its midspan.
    """
    eq_network = fdm(arch_network)

    heights = jnp.array(eq_network.nodes_coordinates())[:, 2]

    assert jnp.allclose(heights, jnp.flip(heights), atol=1e-9)


def test_arch_coordinates_baseline(arch_network):
    """
    The solved node coordinates reproduce the captured baseline.
    """
    eq_network = fdm(arch_network)

    coordinates = jnp.array(eq_network.nodes_coordinates()).tolist()

    assert_baseline("arch_coordinates", coordinates)
