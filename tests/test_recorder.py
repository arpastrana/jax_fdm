"""
Characterization tests for optimization recorder serialization.
"""

import jax.numpy as jnp

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import EdgeForceDensityParameter
from jax_fdm.optimization import OptimizationRecorder


def _record_optimization(network):
    """
    Run a short optimization while recording the parameter history.
    """
    optimizer = SLSQP()
    recorder = OptimizationRecorder(optimizer)
    goals = [EdgeLengthGoal(edge, target=0.6) for edge in network.edges()]
    parameters = [
        EdgeForceDensityParameter(edge, -10.0, -0.1) for edge in network.edges()
    ]

    constrained_fdm(
        network,
        optimizer=optimizer,
        loss=Loss(SquaredError(goals=goals)),
        parameters=parameters,
        maxiter=20,
        callback=recorder,
    )

    return recorder


def test_recorder_json_roundtrip(arch_network, tmp_path):
    """
    A recorder serialized to JSON restores an identical parameter history,
    including the nested node, edge, and face loads.
    """
    recorder = _record_optimization(arch_network)
    filepath = tmp_path / "recorder.json"
    recorder.to_json(str(filepath))

    restored = OptimizationRecorder.from_json(str(filepath))

    history = recorder.history
    history_restored = restored.history

    assert jnp.allclose(jnp.array(history.q), jnp.array(history_restored.q))
    assert jnp.allclose(
        jnp.array(history.xyz_fixed),
        jnp.array(history_restored.xyz_fixed),
    )
    assert jnp.allclose(
        jnp.array(history.loads.nodes),
        jnp.array(history_restored.loads.nodes),
    )
    assert jnp.allclose(
        jnp.array(history.loads.edges),
        jnp.array(history_restored.loads.edges),
    )
    assert jnp.allclose(
        jnp.array(history.loads.faces),
        jnp.array(history_restored.loads.faces),
    )
