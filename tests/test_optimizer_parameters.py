"""
Characterization tests for optimization parameters: loads, supports, and groups.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import NodesColinearGoal
from jax_fdm.goals import NodeXCoordinateGoal
from jax_fdm.goals import NodeYCoordinateGoal
from jax_fdm.goals import NodeZCoordinateGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeGroupForceDensityParameter
from jax_fdm.parameters import NodeGroupLoadXParameter
from jax_fdm.parameters import NodeGroupLoadYParameter
from jax_fdm.parameters import NodeGroupLoadZParameter
from jax_fdm.parameters import NodeGroupSupportXParameter
from jax_fdm.parameters import NodeGroupSupportYParameter
from jax_fdm.parameters import NodeGroupSupportZParameter
from jax_fdm.parameters import NodeLoadZParameter

COORDINATE_GOAL = {0: NodeXCoordinateGoal, 1: NodeYCoordinateGoal, 2: NodeZCoordinateGoal}
SUPPORT_GROUP = {0: NodeGroupSupportXParameter,
                 1: NodeGroupSupportYParameter,
                 2: NodeGroupSupportZParameter}
LOAD_GROUP = {0: NodeGroupLoadXParameter,
              1: NodeGroupLoadYParameter,
              2: NodeGroupLoadZParameter}

AXES = [0, 1, 2]


# ==============================================================================
# Load parameters with a colinearity goal
# ==============================================================================

def _optimize_colinear(network):
    """
    Flatten an arch by tuning the vertical load at every free node.
    """
    goals = [NodesColinearGoal(key=sorted(network.nodes()), weight=1.0)]
    loss = Loss(PredictionError(goals, name="NodesColinearGoal"))
    parameters = [NodeLoadZParameter(node) for node in network.nodes_free()]

    return constrained_fdm(network,
                           optimizer=LBFGSB(),
                           loss=loss,
                           parameters=parameters,
                           maxiter=5000,
                           tol=1e-13)


def test_colinear_load_optimization_flattens_arch(arch_network):
    """
    Optimizing the nodal loads drives every node onto the support line.
    """
    optimized = _optimize_colinear(arch_network)

    heights = jnp.array(optimized.nodes_coordinates())[:, 2]

    assert jnp.allclose(heights, 0.0, atol=1e-9)


def test_colinear_load_optimization_reduces_loss(arch_network):
    """
    The colinearity error after optimization is below the initial solve.
    """
    coordinates_before = jnp.array(fdm(arch_network).nodes_coordinates())[:, 2]
    coordinates_after = jnp.array(_optimize_colinear(arch_network).nodes_coordinates())[:, 2]

    assert jnp.sum(coordinates_after ** 2) < jnp.sum(coordinates_before ** 2)


# ==============================================================================
# Grouped parameters
# ==============================================================================

@pytest.mark.parametrize("axis", AXES)
def test_support_group_shares_single_coordinate(arch_network, axis):
    """
    A grouped support parameter moves every support to a single shared coordinate.
    """
    supports = list(arch_network.nodes_supports())
    goal = COORDINATE_GOAL[axis](key=supports[0], target=0.7)
    parameters = [SUPPORT_GROUP[axis](supports)]

    optimized = constrained_fdm(arch_network,
                                optimizer=LBFGSB(),
                                loss=Loss(SquaredError([goal])),
                                parameters=parameters)

    coordinates = jnp.array([optimized.node_coordinates(node)[axis] for node in supports])

    assert jnp.allclose(coordinates, coordinates[0])
    assert jnp.allclose(coordinates, 0.7, atol=1e-9)


@pytest.mark.parametrize("axis", AXES)
def test_load_group_shares_single_component(arch_network, axis):
    """
    A grouped load parameter tunes one shared load component and leaves the rest.
    """
    free = list(arch_network.nodes_free())
    loads_input = np.array([arch_network.node_load(node) for node in free])
    goal = COORDINATE_GOAL[axis](key=free[len(free) // 2], target=0.3)
    parameters = [LOAD_GROUP[axis](free, -10.0, 10.0)]

    optimized = constrained_fdm(arch_network,
                                optimizer=LBFGSB(),
                                loss=Loss(SquaredError([goal])),
                                parameters=parameters)

    loads = np.array([optimized.node_load(node) for node in free])
    others = [other for other in AXES if other != axis]

    assert jnp.allclose(jnp.asarray(loads[:, axis]), loads[0, axis])
    assert jnp.allclose(jnp.asarray(loads[:, others]), jnp.asarray(loads_input[:, others]))


def test_edge_group_shares_single_forcedensity(arch_network):
    """
    A grouped edge force density parameter collapses every edge to one value.
    """
    edges = list(arch_network.edges())
    goal = NodeZCoordinateGoal(key=list(arch_network.nodes_free())[0], target=-0.5)
    parameters = [EdgeGroupForceDensityParameter(edges, -20.0, -0.01)]

    optimized = constrained_fdm(arch_network,
                                optimizer=LBFGSB(),
                                loss=Loss(SquaredError([goal])),
                                parameters=parameters)

    forcedensities = jnp.array([optimized.edge_forcedensity(edge) for edge in edges])

    assert jnp.allclose(forcedensities, forcedensities[0])
