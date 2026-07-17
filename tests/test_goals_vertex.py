"""
Tests for the vertex goal and constraint families.

The vertex classes are thin retargeted subclasses of their node counterparts:
the quantity logic is inherited unchanged and only the key-to-index resolution
switches from `node_index` to `vertex_index`. These tests pin (1) that the
resolution really goes through the vertex vocabulary, (2) that the inherited
predictions and goal projections stay numerically identical to computing the
quantities directly, (3) that the optimizer's `Collection` machinery keeps node
and vertex classes apart, and (4) that borrowing a Node* goal or constraint on
a mesh structure — which silently worked when both index dicts coincided — is
now a `TypeError` in both directions.
"""

import jax.numpy as jnp
import pytest

from jax_fdm import DTYPE_JAX
from jax_fdm.constraints import NodeXCoordinateConstraint
from jax_fdm.constraints import VertexCurvatureConstraint
from jax_fdm.constraints import VertexXCoordinateConstraint
from jax_fdm.constraints import VertexYCoordinateConstraint
from jax_fdm.constraints import VertexZCoordinateConstraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium.fdm import model_from_sparsity
from jax_fdm.geometry import closest_point_on_line
from jax_fdm.geometry import closest_point_on_plane
from jax_fdm.geometry import closest_point_on_segment
from jax_fdm.geometry import curvature_point_polygon
from jax_fdm.geometry import length_vector
from jax_fdm.goals import NodePointGoal
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals import VertexLineGoal
from jax_fdm.goals import VertexPlaneGoal
from jax_fdm.goals import VertexPointGoal
from jax_fdm.goals import VertexResidualForceGoal
from jax_fdm.goals import VertexResidualVectorGoal
from jax_fdm.goals import VertexSegmentGoal
from jax_fdm.goals import VertexXCoordinateGoal
from jax_fdm.goals import VertexYCoordinateGoal
from jax_fdm.goals import VertexZCoordinateGoal
from jax_fdm.goals import VerticesColinearGoal
from jax_fdm.goals.vertex import VertexGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import EdgeForceDensityParameter
from jax_fdm.optimization.collections import Collection
from jax_fdm.optimization.collections import collect_goals

MAXITER = 50


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


def _fixed_mesh_state(mesh):
    """
    Freeze a mesh's geometry and evaluate one equilibrium state on it.

    Every vertex is anchored, so the equilibrium state reproduces the input
    coordinates exactly and goal predictions reflect the mesh as authored.
    Returns the model, the mesh structure, and the equilibrium state.
    """
    for vkey in mesh.vertices():
        mesh.vertex_support(vkey)
    for edge in mesh.edges():
        mesh.edge_forcedensity(edge, -1.0)

    structure = EquilibriumMeshStructure.from_mesh(mesh)
    model = _model()
    parameters = EquilibriumParametersState.from_datastructure(mesh, dtype=DTYPE_JAX)
    eqstate = model(parameters, structure)

    return model, structure, eqstate


def _anchored_mesh(mesh):
    """
    Anchor a mesh's boundary vertices, load its interior, and set force densities.

    Returns the mesh and the list of free (interior) vertex keys.
    """
    free = []
    for vkey in mesh.vertices():
        if mesh.is_vertex_on_boundary(vkey):
            mesh.vertex_support(vkey)
        else:
            mesh.vertex_load(vkey, [0.0, 0.0, -0.1])
            free.append(vkey)

    for edge in mesh.edges():
        mesh.edge_forcedensity(edge, -1.0)

    return mesh, free


def _vertex_strip(mesh):
    """
    An ordered strip of mesh vertices sharing one x coordinate, sorted by y.
    """
    x_target = mesh.vertex_coordinates(next(iter(mesh.vertices())))[0]
    strip = [
        vkey
        for vkey in mesh.vertices()
        if abs(mesh.vertex_coordinates(vkey)[0] - x_target) < 1e-9
    ]

    return sorted(strip, key=lambda vkey: mesh.vertex_coordinates(vkey)[1])


# ==============================================================================
# Index resolution goes through the vertex vocabulary
# ==============================================================================


def test_vertex_goal_resolves_vertex_index(meshgrid_mesh):
    """
    A vertex goal's index comes from the structure's `vertex_index` mapping.
    """
    model, structure, _ = _fixed_mesh_state(meshgrid_mesh)
    vkey = list(meshgrid_mesh.vertices())[3]

    goal = VertexPointGoal(vkey, target=[0.0, 0.0, 0.0])
    goal.init(model, structure)

    assert int(goal.index[0]) == structure.vertex_index[vkey]


# ==============================================================================
# Inherited predictions and projections are numerically unchanged
# ==============================================================================


def test_vertex_point_and_coordinate_predictions(meshgrid_mesh):
    """
    Point and coordinate goal predictions match direct equilibrium-state slices.
    """
    model, structure, eqstate = _fixed_mesh_state(meshgrid_mesh)
    vkey = list(meshgrid_mesh.vertices())[7]
    index = structure.vertex_index[vkey]

    goal = VertexPointGoal(vkey, target=[0.0, 0.0, 0.0])
    goal.init(model, structure)
    assert jnp.allclose(goal(eqstate).prediction, eqstate.xyz[index, :])

    for goal_cls, axis in [
        (VertexXCoordinateGoal, 0),
        (VertexYCoordinateGoal, 1),
        (VertexZCoordinateGoal, 2),
    ]:
        goal = goal_cls(vkey, target=0.0)
        goal.init(model, structure)
        prediction = goal(eqstate).prediction
        assert jnp.allclose(prediction.ravel(), eqstate.xyz[index, axis])


def test_vertex_residual_predictions(meshgrid_mesh):
    """
    Residual goal predictions match direct equilibrium-state slices.
    """
    model, structure, eqstate = _fixed_mesh_state(meshgrid_mesh)
    vkey = list(meshgrid_mesh.vertices())[5]
    index = structure.vertex_index[vkey]
    residual = eqstate.residuals[index, :]

    goal = VertexResidualVectorGoal(vkey, target=[0.0, 0.0, 0.0])
    goal.init(model, structure)
    assert jnp.allclose(goal(eqstate).prediction, residual)

    goal = VertexResidualForceGoal(vkey, target=0.0)
    goal.init(model, structure)
    assert jnp.allclose(goal(eqstate).prediction.ravel(), length_vector(residual))


def test_vertex_projection_goals(meshgrid_mesh):
    """
    Line, segment, and plane goals project onto their targets exactly.
    """
    model, structure, eqstate = _fixed_mesh_state(meshgrid_mesh)
    vkey = list(meshgrid_mesh.vertices())[11]
    index = structure.vertex_index[vkey]
    point = eqstate.xyz[index, :]

    line = [[0.5, 0.5, -1.0], [0.5, 0.5, 1.0]]
    plane = [[0.0, 0.0, 0.5], [0.0, 0.0, 1.0]]

    for goal_cls, target, closest_fn in [
        (VertexLineGoal, line, closest_point_on_line),
        (VertexSegmentGoal, line, closest_point_on_segment),
        (VertexPlaneGoal, plane, closest_point_on_plane),
    ]:
        goal = goal_cls(vkey, target=target)
        goal.init(model, structure)
        gstate = goal(eqstate)
        expected = closest_fn(point, jnp.asarray(target))
        assert jnp.allclose(gstate.goal, expected)
        assert jnp.allclose(gstate.prediction, point)


# ==============================================================================
# Optimization
# ==============================================================================


def test_vertex_line_goal_optimizes(meshgrid_mesh):
    """
    Constrained form finding with vertex line goals runs to completion.

    Mirrors the pillow example's horizontal projection goal: every free vertex
    is pulled onto the vertical line through its starting position.
    """
    mesh, free = _anchored_mesh(meshgrid_mesh)

    goals = []
    for vkey in free:
        x, y, z = mesh.vertex_coordinates(vkey)
        line = [[x, y, z], [x, y, z + 1.0]]
        goals.append(VertexLineGoal(vkey, target=line))

    loss = Loss(SquaredError(goals=goals))
    parameters = [EdgeForceDensityParameter(edge, -10.0, -0.1) for edge in mesh.edges()]

    optimized = constrained_fdm(
        mesh,
        optimizer=LBFGSB(),
        loss=loss,
        parameters=parameters,
        maxiter=MAXITER,
    )

    residuals = jnp.array(optimized.vertices_residual(keys=free))
    assert jnp.allclose(residuals, 0.0, atol=1e-9)


def test_vertices_colinear_goal_initializes_and_optimizes(meshgrid_mesh):
    """
    An aggregate vertex goal resolves a two-dimensional index and optimizes.

    Regression for the aggregate `init`: it must dispatch `index_from_model`
    through the instance's MRO (vertex resolution), not hop over it to the
    node base.
    """
    mesh, _ = _anchored_mesh(meshgrid_mesh)
    strip = _vertex_strip(mesh)
    assert len(strip) > 2

    goal = VerticesColinearGoal(key=strip)
    model, structure, eqstate = _fixed_mesh_state(mesh.copy())
    goal.init(model, structure)

    assert goal.index.ndim == 2
    expected = tuple(structure.vertex_index[vkey] for vkey in strip)
    assert tuple(int(i) for i in goal.index[0]) == expected

    prediction = goal(eqstate).prediction
    assert jnp.all(jnp.isfinite(prediction))

    loss = Loss(SquaredError(goals=[VerticesColinearGoal(key=strip)]))
    parameters = [EdgeForceDensityParameter(edge, -10.0, -0.1) for edge in mesh.edges()]
    optimized = constrained_fdm(
        mesh,
        optimizer=LBFGSB(),
        loss=loss,
        parameters=parameters,
        maxiter=MAXITER,
    )

    xyz = jnp.array([optimized.vertex_coordinates(vkey) for vkey in strip])
    assert jnp.all(jnp.isfinite(xyz))


# ==============================================================================
# Collection machinery keeps node and vertex classes apart
# ==============================================================================


def test_collect_goals_separates_node_and_vertex_goals():
    """
    Node and vertex point goals form two homogeneous collections.
    """
    goals = [
        NodePointGoal(0, target=[0.0, 0.0, 0.0]),
        VertexPointGoal(1, target=[0.0, 0.0, 0.0]),
        VertexPointGoal(2, target=[0.0, 0.0, 0.0]),
    ]

    collections = collect_goals(goals)

    assert len(collections) == 2
    assert {type(c) for c in collections} == {NodePointGoal, VertexPointGoal}


# ==============================================================================
# Vocabulary enforcement: node <-> vertex borrowing is an error
# ==============================================================================


def test_node_goal_on_mesh_raises(meshgrid_mesh):
    """
    A node goal refuses to initialize against a mesh structure.
    """
    model, structure, _ = _fixed_mesh_state(meshgrid_mesh)
    goal = NodePointGoal(0, target=[0.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="Vertex"):
        goal.init(model, structure)


def test_vertex_goal_on_network_raises(arch_network):
    """
    A vertex goal refuses to initialize against a network structure.
    """
    structure = EquilibriumStructure.from_network(arch_network)
    model = _model()
    goal = VertexPointGoal(0, target=[0.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="Node"):
        goal.init(model, structure)


def test_node_goal_on_network_still_works(arch_network):
    """
    Node goals keep resolving on network structures.
    """
    structure = EquilibriumStructure.from_network(arch_network)
    model = _model()
    node = next(iter(arch_network.nodes()))

    goal = NodePointGoal(node, target=[0.0, 0.0, 0.0])
    goal.init(model, structure)

    assert int(goal.index[0]) == structure.node_index[node]


def test_node_constraint_on_mesh_raises(meshgrid_mesh):
    """
    A node constraint refuses to initialize against a mesh structure.
    """
    model, structure, _ = _fixed_mesh_state(meshgrid_mesh)
    constraint = NodeXCoordinateConstraint(0, bound_low=0.0, bound_up=1.0)

    with pytest.raises(TypeError, match="Vertex"):
        constraint.init(model, structure)


# ==============================================================================
# Vertex constraints
# ==============================================================================


def test_vertex_coordinate_constraints(meshgrid_mesh):
    """
    The rebased coordinate constraints still slice the right coordinate.
    """
    model, structure, eqstate = _fixed_mesh_state(meshgrid_mesh)
    vkey = list(meshgrid_mesh.vertices())[4]
    index = structure.vertex_index[vkey]

    for constraint_cls, axis in [
        (VertexXCoordinateConstraint, 0),
        (VertexYCoordinateConstraint, 1),
        (VertexZCoordinateConstraint, 2),
    ]:
        constraint = constraint_cls(vkey, bound_low=-1.0, bound_up=1.0)
        constraint.init(model, structure)
        assert int(constraint.index[0]) == index
        value = constraint.constraint(eqstate, constraint.index[0])
        assert jnp.allclose(value, eqstate.xyz[index, axis])


def test_vertex_curvature_constraint(meshgrid_mesh):
    """
    The curvature constraint resolves its polygon through the vertex vocabulary
    and reproduces the discrete curvature computed directly.
    """
    model, structure, eqstate = _fixed_mesh_state(meshgrid_mesh)
    interior = [
        vkey
        for vkey in meshgrid_mesh.vertices()
        if not meshgrid_mesh.is_vertex_on_boundary(vkey)
    ]
    vkey = interior[0]
    polygon = meshgrid_mesh.vertex_neighbors(vkey, ordered=True)

    constraint = VertexCurvatureConstraint(
        vkey,
        polygon,
        bound_low=-1.0,
        bound_up=1.0,
    )
    constraint.init(model, structure)

    index = structure.vertex_index[vkey]
    assert int(constraint.index[0]) == index

    index_polygon = constraint.index_polygon[index, :]
    expected_polygon = tuple(structure.vertex_index[nbr] for nbr in polygon)
    assert tuple(int(i) for i in index_polygon) == expected_polygon

    value = constraint.constraint(eqstate, constraint.index[0])
    expected = curvature_point_polygon(
        eqstate.xyz[index, :],
        eqstate.xyz[index_polygon, :],
    )
    assert jnp.allclose(value, expected)


# ==============================================================================
# Contract-teaching error messages
# ==============================================================================


def test_collection_names_missing_attribute():
    """
    A goal whose init parameter is not stored under the same name gets a
    teaching error from the collection machinery.
    """

    class ForgetfulPointGoal(NodePointGoal):
        def __init__(self, key, target, stiffness, weight=1.0):
            super().__init__(key, target, weight)
            self._stiffness = stiffness

    goals = [
        ForgetfulPointGoal(0, [0.0, 0.0, 0.0], stiffness=1.0),
        ForgetfulPointGoal(1, [0.0, 0.0, 0.0], stiffness=2.0),
    ]

    with pytest.raises(AttributeError, match="stored as attribute 'self.stiffness'"):
        Collection(goals)


def test_goal_shape_mismatch_names_the_fix(meshgrid_mesh):
    """
    A scalar prediction that drops its trailing axis gets a teaching error.
    """

    class SquashedZGoal(ScalarGoal, VertexGoal):
        def prediction(self, eq_state, index):
            # wrong on purpose: shape () instead of (1,)
            return eq_state.xyz[index, 2]

    model, structure, eqstate = _fixed_mesh_state(meshgrid_mesh)
    goal = SquashedZGoal(0, target=0.0)
    goal.init(model, structure)

    with pytest.raises(ValueError, match="atleast_1d"):
        goal(eqstate)
