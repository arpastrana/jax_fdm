"""
Tests for the `evaluate` family: goals, constraints, errors, losses, and
parameters evaluated straight off a datastructure, with no optimization.

`evaluate` reads the equilibrium state, structure, and parameters off the
high-level COMPAS layer, then reads the quantity of interest off the
datastructure's current geometry as-is (no form-finding). These tests pin that
the shortcut reproduces the manual pipeline, keeps dense and sparse in parity,
and never mutates the goal or constraint it runs on, which is the regression the
stateless refactor removed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fdm.constraints import EdgeAngleConstraint
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import DatastructureState
from jax_fdm.equilibrium import datastructure_state
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeAngleGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import EdgesLengthEqualGoal
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.losses import L2Regularizer
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.losses import PredictionError
from jax_fdm.losses import SquaredError
from jax_fdm.parameters import EdgeForceDensityParameter


@pytest.fixture
def arch():
    """
    A small form-found arch network with supports, loads, and force densities.
    """
    network = FDNetwork()
    for k in range(5):
        network.add_node(k, x=float(k), y=0.0, z=0.0)
    for edge in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        network.add_edge(*edge)
    network.node_support(0)
    network.node_support(4)
    network.nodes_loads([0.0, 0.0, -1.0], keys=[1, 2, 3])
    network.edges_forcedensities(-1.5)

    return fdm(network)


# ==============================================================================
# Factory
# ==============================================================================


def test_factory_returns_named_tuple(arch):
    """
    The factory returns a typed bundle whose state matches the datastructure.
    """
    equilibrium = datastructure_state(arch, sparse=False)

    assert isinstance(equilibrium, DatastructureState)

    xyz = jnp.asarray(arch.nodes_coordinates())
    assert jnp.allclose(equilibrium.eq_state.xyz, xyz)


def test_factory_dense_sparse_parity(arch):
    """
    The dense and sparse factories assemble the same equilibrium state.
    """
    dense = datastructure_state(arch, sparse=False)
    sparse = datastructure_state(arch, sparse=True)

    assert jnp.allclose(dense.eq_state.lengths, sparse.eq_state.lengths)
    assert jnp.allclose(dense.eq_state.forces, sparse.eq_state.forces)
    assert jnp.allclose(dense.eq_state.residuals, sparse.eq_state.residuals)


def test_factory_state_mirrors_stored_attributes(arch):
    """
    The state is read off the datastructure's stored attributes, not recomputed.
    """
    eq_state = datastructure_state(arch, sparse=False).eq_state

    lengths = jnp.reshape(jnp.asarray(arch.edges_lengths()), (-1, 1))
    forces = jnp.reshape(jnp.asarray(arch.edges_forces()), (-1, 1))

    assert jnp.allclose(eq_state.xyz, jnp.asarray(arch.nodes_coordinates()))
    assert jnp.allclose(eq_state.lengths, lengths)
    assert jnp.allclose(eq_state.forces, forces)
    assert jnp.allclose(eq_state.residuals, jnp.asarray(arch.nodes_residual()))


def test_factory_fresh_datastructure_reads_zero_state():
    """
    A never-solved datastructure reads back zero lengths, forces, and residuals.

    The factory reflects stored attributes rather than form-finding, so the
    edge and residual defaults (0.0) survive until ``fdm`` writes a solved state.
    The edge vectors still follow from the node coordinates, and so stay nonzero.
    """
    network = FDNetwork()
    for k in range(3):
        network.add_node(k, x=float(k), y=0.0, z=0.0)
    for edge in [(0, 1), (1, 2)]:
        network.add_edge(*edge)
    network.node_support(0)
    network.node_support(2)
    network.edges_forcedensities(-1.0)

    eq_state = datastructure_state(network, sparse=False).eq_state

    assert jnp.allclose(eq_state.lengths, 0.0)
    assert jnp.allclose(eq_state.forces, 0.0)
    assert jnp.allclose(eq_state.residuals, 0.0)
    assert not jnp.allclose(eq_state.vectors, 0.0)


# ==============================================================================
# Goal.evaluate
# ==============================================================================


def test_goal_evaluate_matches_manual_length(arch):
    """
    A length goal's evaluated prediction equals the datastructure's edge length.
    """
    edge = (1, 2)
    goal = EdgeLengthGoal(edge, 2.0)

    gstate = goal.evaluate(arch, sparse=False)

    assert jnp.allclose(gstate.prediction.ravel()[0], arch.edge_length(edge))


def test_goal_evaluate_dense_sparse_parity(arch):
    """
    A goal evaluates identically on the dense and sparse pipelines.
    """
    goal = NodeResidualForceGoal(2, 0.0)

    dense = goal.evaluate(arch, sparse=False)
    sparse = goal.evaluate(arch, sparse=True)

    assert jnp.allclose(dense.prediction, sparse.prediction)


def test_vector_goal_evaluate_leaves_original_unmutated(arch):
    """
    Evaluating a vector goal does not mutate its reference vector.

    Regression for the stateful `init`, which scattered `self.vector` into a
    per-index table on the user's own object; a second evaluation then read the
    corrupted table.
    """
    goal = EdgeAngleGoal((1, 2), [0.0, 0.0, 1.0], 0.0)
    assert goal.vector.shape == (1, 3)

    first = goal.evaluate(arch, sparse=False)
    assert goal.vector.shape == (1, 3)

    second = goal.evaluate(arch, sparse=False)
    assert goal.vector.shape == (1, 3)
    assert jnp.allclose(first.prediction, second.prediction)


def test_aggregate_goal_evaluate(arch):
    """
    An aggregate goal reduces over its whole key list in one evaluation.
    """
    goal = EdgesLengthEqualGoal([(0, 1), (1, 2), (2, 3), (3, 4)])

    gstate = goal.evaluate(arch, sparse=False)

    assert gstate.prediction.shape == (1, 1)
    assert jnp.all(jnp.isfinite(gstate.prediction))


# ==============================================================================
# Constraint.evaluate
# ==============================================================================


def test_constraint_evaluate_matches_manual_length(arch):
    """
    A length constraint's evaluated value equals the datastructure's edge length.
    """
    edge = (2, 3)
    constraint = EdgeLengthConstraint(edge, 0.5, 2.0)

    value = constraint.evaluate(arch, sparse=False)

    assert jnp.allclose(value.ravel()[0], arch.edge_length(edge))


def test_edge_angle_constraint_evaluate_leaves_original_unmutated(arch):
    """
    Evaluating a vector constraint does not mutate its reference vector.
    """
    constraint = EdgeAngleConstraint((1, 2), [0.0, 0.0, 1.0], 0.0, 1.0)
    assert constraint.vector.shape == (1, 3)

    first = constraint.evaluate(arch, sparse=False)
    assert constraint.vector.shape == (1, 3)

    second = constraint.evaluate(arch, sparse=False)
    assert jnp.allclose(first, second)


# ==============================================================================
# Error.evaluate and Loss.evaluate
# ==============================================================================


def test_error_evaluate_is_finite(arch):
    """
    An error term evaluates to a finite scalar over its raw goals.
    """
    goals = [EdgeLengthGoal(edge, 1.0) for edge in arch.edges()]
    error = SquaredError(goals)

    value = error.evaluate(arch, sparse=False)

    assert value.shape == ()
    assert jnp.isfinite(value)


def test_mean_squared_error_divides_by_goal_count(arch):
    """
    A mean-style error term's evaluation divides the summed error by the goal
    count, matching its optimization-time normalization.
    """
    goals = [EdgeLengthGoal(edge, 1.0) for edge in arch.edges()]

    summed = SquaredError(goals).evaluate(arch, sparse=False)
    mean = MeanSquaredError(goals).evaluate(arch, sparse=False)

    assert jnp.allclose(mean, summed / len(goals))


def test_loss_evaluate_sums_terms_and_regularizers(arch):
    """
    A loss evaluation sums its error terms and regularizers on one equilibrium.
    """
    goals = [EdgeLengthGoal(edge, 1.0) for edge in arch.edges()]
    error = SquaredError(goals)
    loadpath = PredictionError([NetworkLoadPathGoal()], alpha=0.01)
    regularizer = L2Regularizer(alpha=0.1)

    loss = Loss(error, loadpath, regularizer)
    value = loss.evaluate(arch, sparse=False)

    equilibrium = datastructure_state(arch, sparse=False)
    expected = (
        error.evaluate(arch, sparse=False)
        + loadpath.evaluate(arch, sparse=False)
        + regularizer(equilibrium.parameters)
    )

    assert jnp.allclose(value, expected)


def test_loss_evaluate_dense_sparse_parity(arch):
    """
    A loss evaluates to the same value on the dense and sparse pipelines.
    """
    goals = [EdgeLengthGoal(edge, 1.0) for edge in arch.edges()]
    loss = Loss(SquaredError(goals), L2Regularizer(alpha=0.1))

    dense = loss.evaluate(arch, sparse=False)
    sparse = loss.evaluate(arch, sparse=True)

    assert jnp.allclose(dense, sparse, atol=1e-5)


# ==============================================================================
# Parameter.evaluate
# ==============================================================================


def test_parameter_evaluate_matches_value(arch):
    """
    A parameter's evaluation aliases its `value`, reading straight off the edge.
    """
    edge = (1, 2)
    parameter = EdgeForceDensityParameter(edge)

    assert parameter.evaluate(arch) == parameter.value(arch)
    assert np.allclose(parameter.evaluate(arch), -1.5)
