"""
Tests for the residual plane goals.

The goal drives the residual force vector at a support to lie in a target
plane through the origin, described by its normal vector. These tests pin
(1) the projection math: the prediction is the unit residual direction, the
goal is its in-plane projection, and the error term is the out-of-plane
component whose norm is the sine of the angle to the plane; (2) that the
error is invariant to the residual's magnitude and to the normal's scale;
(3) that the vertex counterpart resolves through the vertex vocabulary; and
(4) that optimizing the goal on an arch rotates the support reaction into
the target plane.
"""

import jax.numpy as jnp
import pytest

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium.fdm import model_from_sparsity
from jax_fdm.geometry import normalize_vector
from jax_fdm.goals import NodeResidualPlaneGoal
from jax_fdm.goals import VertexResidualPlaneGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter

MAXITER = 100


# ==============================================================================
# Helpers
# ==============================================================================


def _model():
    """
    A dense, single-iteration equilibrium model.
    """
    return model_from_sparsity(
        sparse=False,
        tmax=1,
        eta=1e-6,
        is_load_local=False,
        itersolve_fn=None,
        iterload_fn=None,
        implicit_diff=True,
        verbose=False,
    )


def _network_state(network):
    """
    Evaluate one equilibrium state on a network.

    Returns the model, the network structure, and the equilibrium state.
    """
    structure = EquilibriumStructure.from_network(network)
    model = _model()
    parameters = EquilibriumParametersState.from_datastructure(
        network,
        dtype=DTYPE_JAX,
    )
    eqstate = model(parameters, structure)

    return model, structure, eqstate


# ==============================================================================
# Projection math
# ==============================================================================


def test_residual_plane_goal_projection(arch_network):
    """
    The prediction is the unit residual, the goal its in-plane projection, and
    the error norm the sine of the angle between the residual and the plane.
    """
    model, structure, eqstate = _network_state(arch_network)
    support = 0
    index = structure.node_index[support]
    residual = eqstate.residuals[index, :]

    # a non-unit normal, to exercise the internal normalization
    normal = jnp.asarray([2.0, 0.0, 2.0])

    goal = NodeResidualPlaneGoal(support, target=normal)
    gstate = goal(eqstate, structure)

    direction = normalize_vector(residual)
    assert jnp.allclose(gstate.prediction, direction)

    unit_normal = normalize_vector(normal)
    projection = direction - jnp.dot(direction, unit_normal) * unit_normal
    assert jnp.allclose(gstate.goal, projection)

    # the goal output lies in the plane
    assert jnp.allclose(jnp.dot(jnp.ravel(gstate.goal), unit_normal), 0.0)

    # error norm = |direction . normal| = sine of the angle to the plane
    error = jnp.linalg.norm(gstate.prediction - gstate.goal)
    assert jnp.allclose(error, jnp.abs(jnp.dot(direction, unit_normal)))


def test_residual_plane_goal_invariances(arch_network):
    """
    The error is invariant to the residual magnitude and to the normal's scale,
    and vanishes exactly when the residual lies in the plane.
    """
    model, structure, eqstate = _network_state(arch_network)
    support = 0
    index = structure.node_index[support]
    residual = eqstate.residuals[index, :]

    def error(eqstate, normal):
        goal = NodeResidualPlaneGoal(support, target=normal)
        gstate = goal(eqstate, structure)
        return jnp.sum(jnp.square(gstate.prediction - gstate.goal))

    normal = [0.0, 0.0, 1.0]

    # scaling the residuals leaves the error unchanged
    scaled = eqstate._replace(residuals=eqstate.residuals * 10.0)
    assert jnp.allclose(error(eqstate, normal), error(scaled, normal))

    # scaling the normal leaves the error unchanged
    assert jnp.allclose(error(eqstate, normal), error(eqstate, [0.0, 0.0, 7.0]))

    # a plane that contains the residual gives zero error: the residual of the
    # planar arch has no y component, so it lies in the y-normal plane
    assert jnp.allclose(residual[1], 0.0, atol=1e-12)
    assert jnp.allclose(error(eqstate, [0.0, 1.0, 0.0]), 0.0, atol=1e-24)


# ==============================================================================
# Vertex counterpart
# ==============================================================================


def test_vertex_residual_plane_goal(meshgrid_mesh):
    """
    The vertex goal resolves through the vertex vocabulary and inherits the
    node projection unchanged.
    """
    for vkey in meshgrid_mesh.vertices():
        meshgrid_mesh.vertex_support(vkey)
    for edge in meshgrid_mesh.edges():
        meshgrid_mesh.edge_forcedensity(edge, -1.0)

    structure = EquilibriumMeshStructure.from_mesh(meshgrid_mesh)
    model = _model()
    parameters = EquilibriumParametersState.from_datastructure(
        meshgrid_mesh,
        dtype=DTYPE_JAX,
    )
    eqstate = model(parameters, structure)

    vkey = list(meshgrid_mesh.vertices())[3]
    normal = jnp.asarray([0.0, 0.0, 1.0])

    goal = VertexResidualPlaneGoal(vkey, target=normal)

    index = structure.vertex_index[vkey]
    assert goal.index_from_structure(structure) == index

    gstate = goal(eqstate, structure)
    direction = normalize_vector(eqstate.residuals[index, :])
    projection = direction - jnp.dot(direction, normal) * normal
    assert jnp.allclose(gstate.prediction, direction)
    assert jnp.allclose(gstate.goal, projection)


def test_vertex_residual_plane_goal_on_network_raises(arch_network):
    """
    The vertex goal refuses to initialize against a network structure.
    """
    model, structure, _ = _network_state(arch_network)
    goal = VertexResidualPlaneGoal(0, target=[0.0, 0.0, 1.0])

    with pytest.raises(TypeError, match="Node"):
        goal.index_from_structure(structure)


# ==============================================================================
# Optimization
# ==============================================================================


def test_residual_plane_goal_optimizes(arch_network):
    """
    Optimizing the goal rotates the support reactions into the target planes.

    The reaction at an end support of the arch points along its first edge,
    whose slope is the vertical-to-horizontal force ratio: the vertical part
    is fixed by statics, the horizontal part is set by the force densities.
    The target planes demand a slope of 4, which is inside that reachable
    family but far from the initial slope, so the optimizer has to steer the
    force densities to meet it.
    """
    supports = [0, 10]
    normals = {
        supports[0]: jnp.asarray([4.0, 0.0, -1.0]),
        supports[1]: jnp.asarray([-4.0, 0.0, -1.0]),
    }

    _, structure, eqstate = _network_state(arch_network)
    for support in supports:
        index = structure.node_index[support]
        direction = normalize_vector(eqstate.residuals[index, :])
        sine = jnp.dot(direction, normalize_vector(normals[support]))
        assert jnp.abs(sine) > 0.1  # the goal starts unmet

    goals = [
        NodeResidualPlaneGoal(support, target=normals[support]) for support in supports
    ]
    loss = Loss(SquaredError(goals=goals))
    parameters = [
        EdgeForceDensityParameter(edge, -20.0, -0.1) for edge in arch_network.edges()
    ]

    optimized = constrained_fdm(
        arch_network,
        optimizer=LBFGSB(),
        loss=loss,
        parameters=parameters,
        maxiter=MAXITER,
    )

    for support in supports:
        residual = jnp.asarray(optimized.node_residual(support))
        direction = normalize_vector(residual)
        sine = jnp.dot(direction, normalize_vector(normals[support]))
        assert jnp.abs(sine) < 1e-3  # the reaction lies in the plane
