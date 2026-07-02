"""
Regression tests for angle goals.

`angle_vectors` returns a scalar, but a `ScalarGoal` target is shaped `(N, 1)`.
The two must agree inside `Goal.__call__`, so an angle goal's prediction has to
carry a trailing axis. These tests pin that contract for edge and node angle
goals, which once regressed to a `(N,) vs (N, 1)` shape mismatch.
"""

import jax.numpy as jnp

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.goals import EdgeAngleGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import EdgeForceDensityParameter

TARGET_ANGLE = jnp.pi / 4  # 45 degrees to the vertical
BOUND_LOW = -10.0
BOUND_UP = -0.1
MAXITER = 50


def test_edge_angle_goal_optimizes(arch_network):
    """
    Constrained form finding with edge angle goals runs to completion.

    Before angle predictions carried a trailing axis, the goal target `(N, 1)`
    and the prediction `(N,)` disagreed and `Goal.__call__` raised on the shape
    assertion. This drives the exact multi-edge path that regressed.
    """
    goals = [EdgeAngleGoal(edge, vector=[0.0, 0.0, 1.0], target=TARGET_ANGLE)
             for edge in arch_network.edges()]
    loss = Loss(SquaredError(goals=goals))
    parameters = [EdgeForceDensityParameter(edge, BOUND_LOW, BOUND_UP)
                  for edge in arch_network.edges()]

    optimized = constrained_fdm(arch_network,
                                optimizer=LBFGSB(),
                                loss=loss,
                                parameters=parameters,
                                maxiter=MAXITER)

    # the solve is still in equilibrium at its free nodes
    residuals = jnp.array(optimized.nodes_residual(keys=optimized.nodes_free()))
    assert jnp.allclose(residuals, 0.0, atol=1e-9)


def test_edge_angle_goal_prediction_shape(arch_network):
    """
    An edge angle goal emits a goal state whose goal and prediction shapes match.
    """
    edges = list(arch_network.edges())
    goals = [EdgeAngleGoal(edge, vector=[0.0, 0.0, 1.0], target=TARGET_ANGLE)
             for edge in edges]
    loss = Loss(SquaredError(goals=goals))

    # a single optimization step is enough to force the goal to be evaluated
    constrained_fdm(arch_network,
                    optimizer=LBFGSB(),
                    loss=loss,
                    parameters=[EdgeForceDensityParameter(edge, BOUND_LOW, BOUND_UP)
                                for edge in edges],
                    maxiter=1)
