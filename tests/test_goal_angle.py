"""
Regression tests for angle goals.

`angle_vectors` returns a scalar, but a `ScalarGoal` target is shaped `(N, 1)`.
The two must agree inside `Goal.__call__`, so an angle goal's prediction has to
carry a trailing axis. These tests pin that contract for edge and vertex angle
goals, which once regressed to a `(N,) vs (N, 1)` shape mismatch.

The vertex angle goals additionally compute a vertex normal from the faces
surrounding a vertex. That normal averages the incident face normals, and on a
ragged mesh (mixed triangular and quad faces) the shorter face rows in
`faces_indexed` are `-1`-padded. The padding is replaced by a duplicated vertex
so it contributes a zero-area term to the Newell sum and never introduces a
`nan`, in the value or in the gradient. These tests pin both the geometric
correctness of that normal and its `nan`-reliability, and they run each check on
a regular quad grid and on a `ragged_mesh` so the padding path is exercised
against a padding-free baseline.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium.fdm import model_from_sparsity
from jax_fdm.goals import EdgeAngleGoal
from jax_fdm.goals import VertexNormalAngleGoal
from jax_fdm.goals import VertexTangentAngleGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanAbsoluteError
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


# ==============================================================================
# Vertex angle goals (mesh-only)
# ==============================================================================

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


def _predictions_on_fixed_mesh(mesh, goal_cls, keys, vector, target=0.0):
    """
    Evaluate a vertex angle goal directly on a mesh whose geometry is frozen.

    Every vertex is anchored, so the equilibrium state reproduces the input
    coordinates exactly and the goal prediction reflects the mesh as authored,
    independent of any optimizer. Returns the per-key predictions in radians.
    """
    for vkey in mesh.vertices():
        mesh.vertex_support(vkey)
    for edge in mesh.edges():
        mesh.edge_forcedensity(edge, -1.0)

    structure = EquilibriumMeshStructure.from_mesh(mesh)
    model = model_from_sparsity(sparse=False,
                                tmax=1,
                                eta=1e-6,
                                is_load_local=False,
                                itersolve_fn=None,
                                iterload_fn=None,
                                implicit_diff=True,
                                verbose=False)
    parameters = EquilibriumParametersState.from_datastructure(mesh, dtype=DTYPE_JAX)
    eqstate = model(parameters, structure)

    xyz_in = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
    assert jnp.allclose(eqstate.xyz, xyz_in), "anchored mesh must preserve geometry"

    goals = [goal_cls(vkey, vector=vector, target=target) for vkey in keys]
    predictions = []
    for goal in goals:
        goal.init(model, structure)
        predictions.append(float(goal(eqstate).prediction.ravel()[0]))

    return jnp.asarray(predictions)


def _tilt_onto_plane(mesh, theta):
    """
    Lift every vertex of `mesh` onto the plane `z = tan(theta) * x`.

    The whole mesh then shares one normal, inclined `theta` from `[0, 0, 1]`.
    """
    for vkey in mesh.vertices():
        x, _, _ = mesh.vertex_coordinates(vkey)
        mesh.vertex_attribute(vkey, "z", math.tan(theta) * x)

    return mesh


def _triangulated_vertices(mesh):
    """
    The interior vertices of `mesh` that touch a triangular (non-quad) face.
    """
    keys = set()
    for fkey in mesh.faces():
        vertices = mesh.face_vertices(fkey)
        if len(vertices) != 3:
            continue
        keys.update(v for v in vertices if not mesh.is_vertex_on_boundary(v))

    return sorted(keys)


def _loss_grad_wrt_forcedensities(mesh, loss):
    """
    The gradient of `loss` with respect to the mesh force densities.

    Reproduces the optimizer's backward pass without running a solve: the goal
    collections are initialized through the same `Optimizer.goals` entry point,
    and the loss is differentiated with respect to the force-density vector `q`
    (the equilibrium state's only geometry-bearing input here). Returns the
    gradient array.
    """
    structure = EquilibriumMeshStructure.from_mesh(mesh)
    model = model_from_sparsity(sparse=False,
                                tmax=1,
                                eta=1e-6,
                                is_load_local=False,
                                itersolve_fn=None,
                                iterload_fn=None,
                                implicit_diff=True,
                                verbose=False)
    parameters = EquilibriumParametersState.from_datastructure(mesh, dtype=DTYPE_JAX)

    # build the goal collections through the real optimizer code path
    LBFGSB().goals(loss, model, structure)

    def loss_of_forcedensities(q):
        params = EquilibriumParametersState(q=q,
                                            xyz_fixed=parameters.xyz_fixed,
                                            loads=parameters.loads)
        return loss(params, model, structure)

    return jax.grad(loss_of_forcedensities)(parameters.q)


# ------------------------------------------------------------------------------
# Correctness: the vertex normal is geometrically right, quads and triangles alike
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fixture", ["meshgrid_mesh", "ragged_mesh"])
def test_vertex_normal_angle_flat_mesh(request, mesh_fixture):
    """
    On a flat (z = 0) mesh every vertex normal is vertical.

    So the normal makes a 0 rad angle with `[0, 0, 1]`, and this must hold
    identically on the quad grid and on the ragged mesh whose triangular faces
    drive the `-1` padding path. Any padding leak would tilt the averaged normal
    and show up here.
    """
    mesh = request.getfixturevalue(mesh_fixture)
    interior = [v for v in mesh.vertices() if not mesh.is_vertex_on_boundary(v)]

    predictions = _predictions_on_fixed_mesh(mesh, VertexNormalAngleGoal,
                                              interior, vector=[0.0, 0.0, 1.0])

    assert jnp.allclose(predictions, 0.0, atol=1e-6)


@pytest.mark.parametrize("mesh_fixture", ["meshgrid_mesh", "ragged_mesh"])
def test_vertex_tangent_angle_flat_mesh(request, mesh_fixture):
    """
    The tangent angle is 90 deg minus the normal angle, so on a flat mesh it is
    exactly pi / 2 — again identically for the quad and ragged meshes.
    """
    mesh = request.getfixturevalue(mesh_fixture)
    interior = [v for v in mesh.vertices() if not mesh.is_vertex_on_boundary(v)]

    predictions = _predictions_on_fixed_mesh(mesh, VertexTangentAngleGoal,
                                             interior, vector=[0.0, 0.0, 1.0])

    assert jnp.allclose(predictions, jnp.pi / 2, atol=1e-6)


def test_vertex_normal_angle_tilted_plane_at_triangle(ragged_mesh):
    """
    A vertex touching a triangulated face reports the correct normal angle.

    Tilting the ragged mesh onto the plane `z = tan(theta) * x` gives every
    vertex the same normal, inclined `theta` from `[0, 0, 1]`. Checking a vertex
    incident to a triangle proves the padded (triangular) face rows average into
    the same normal as the quad rows — the padding neither biases nor corrupts it.
    """
    theta = math.radians(30.0)
    _tilt_onto_plane(ragged_mesh, theta)

    triangulated = _triangulated_vertices(ragged_mesh)
    assert triangulated, "ragged_mesh must have interior vertices on triangles"

    predictions = _predictions_on_fixed_mesh(ragged_mesh, VertexNormalAngleGoal,
                                             triangulated, vector=[0.0, 0.0, 1.0])

    assert jnp.allclose(predictions, theta, atol=1e-6)


# ------------------------------------------------------------------------------
# Winding-invariance: the angle folds into [0, pi / 2], sign of the normal aside
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("goal_cls", [VertexNormalAngleGoal, VertexTangentAngleGoal])
def test_vertex_angle_invariant_to_face_winding(meshgrid_mesh, goal_cls):
    """
    Reversing every face's winding must not change the predicted angle.

    The vertex normal is the average of the incident face normals, so flipping
    the winding flips that normal's sign. A prediction taken from the *signed*
    cosine would swing by `pi` between the two windings (an inclined normal that
    read `theta` from `[0, 0, 1]` would read `pi - theta` once flipped); folding
    the angle into `[0, pi / 2]` via the absolute cosine keeps both windings
    equal. This is the exact defect that made the pringle tangent goal chase the
    wrong branch and collapse the surface to a zero tangent angle instead of the
    target. The mesh is tilted off horizontal so the two branches are distinct
    (at `theta = 0` they coincide and the test would not discriminate).
    """
    _tilt_onto_plane(meshgrid_mesh, math.radians(30.0))
    interior = [v for v in meshgrid_mesh.vertices()
                if not meshgrid_mesh.is_vertex_on_boundary(v)]

    predictions = _predictions_on_fixed_mesh(meshgrid_mesh, goal_cls,
                                             interior, vector=[0.0, 0.0, 1.0])

    flipped = meshgrid_mesh.copy()
    for fkey in list(flipped.faces()):
        flipped.face_vertices(fkey).reverse()
    predictions_flipped = _predictions_on_fixed_mesh(flipped, goal_cls,
                                                     interior, vector=[0.0, 0.0, 1.0])

    assert jnp.allclose(predictions, predictions_flipped, atol=1e-6)


def test_vertex_normal_angle_in_upper_hemisphere(meshgrid_mesh):
    """
    A downward-pointing vertex normal reports its acute angle, not the obtuse one.

    Tilting the mesh onto `z = tan(theta) * x` and then flipping the winding
    gives every vertex a normal that points below horizontal. The reported angle
    to `[0, 0, 1]` must still be `theta` (in `[0, pi / 2]`), not `pi - theta`.
    """
    theta = math.radians(30.0)
    _tilt_onto_plane(meshgrid_mesh, theta)
    for fkey in list(meshgrid_mesh.faces()):
        meshgrid_mesh.face_vertices(fkey).reverse()

    interior = [v for v in meshgrid_mesh.vertices()
                if not meshgrid_mesh.is_vertex_on_boundary(v)]
    predictions = _predictions_on_fixed_mesh(meshgrid_mesh, VertexNormalAngleGoal,
                                             interior, vector=[0.0, 0.0, 1.0])

    assert jnp.allclose(predictions, theta, atol=1e-6)


# ------------------------------------------------------------------------------
# nan-reliability and shape: optimization drives the padding path in the gradient
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("goal_cls", [VertexNormalAngleGoal, VertexTangentAngleGoal])
def test_vertex_angle_goal_optimizes(meshgrid_mesh, goal_cls):
    """
    Constrained form finding with vertex angle goals runs to completion.

    The vertex normal is averaged from the incident face normals, so this drives
    the mesh face machinery (`faces_indexed` + `connectivity_faces_vertices`) that
    replaced the removed `connectivity_faces`/`face_node_index` accessors.
    """
    mesh, free = _anchored_mesh(meshgrid_mesh)

    goals = [goal_cls(vkey, vector=[0.0, 0.0, 1.0], target=TARGET_ANGLE)
             for vkey in free]
    loss = Loss(SquaredError(goals=goals))
    parameters = [EdgeForceDensityParameter(edge, BOUND_LOW, BOUND_UP)
                  for edge in mesh.edges()]

    optimized = constrained_fdm(mesh,
                                optimizer=LBFGSB(),
                                loss=loss,
                                parameters=parameters,
                                maxiter=MAXITER)

    # the solve is still in equilibrium at its free vertices
    residuals = jnp.array([optimized.vertex_residual(vkey) for vkey in free])
    assert jnp.allclose(residuals, 0.0, atol=1e-9)


@pytest.mark.parametrize("goal_cls", [VertexNormalAngleGoal, VertexTangentAngleGoal])
def test_vertex_angle_goal_ragged_optimizes_nan_free(ragged_mesh, goal_cls):
    """
    Vertex angle goals optimize `nan`-free on a ragged (tri + quad) mesh.

    `ragged_mesh` has triangular faces whose `faces_indexed` rows are `-1`-padded,
    so every one of the L-BFGS-B gradient steps below backpropagates through the
    padding-replacement `jnp.where`. A `nan` from that path — the classic
    "gradient of the duplicated vertex" failure — would poison the loss and abort
    the solve, so a clean run that stays in equilibrium is the reliability check.
    """
    mesh, free = _anchored_mesh(ragged_mesh)

    goals = [goal_cls(vkey, vector=[0.0, 0.0, 1.0], target=TARGET_ANGLE)
             for vkey in free]
    loss = Loss(SquaredError(goals=goals))
    parameters = [EdgeForceDensityParameter(edge, BOUND_LOW, BOUND_UP)
                  for edge in mesh.edges()]

    optimized = constrained_fdm(mesh,
                                optimizer=LBFGSB(),
                                loss=loss,
                                parameters=parameters,
                                maxiter=MAXITER)

    residuals = jnp.array([optimized.vertex_residual(vkey) for vkey in free])
    assert not jnp.any(jnp.isnan(residuals))
    assert jnp.allclose(residuals, 0.0, atol=1e-9)


@pytest.mark.parametrize("mesh_fixture", ["meshgrid_mesh", "ragged_mesh"])
def test_vertex_normal_angle_loss_gradient_is_finite(request, mesh_fixture):
    """
    The gradient of a whole-mesh normal-angle loss is finite (no `nan`s).

    A `VertexNormalAngleGoal` is placed on *every* vertex — including the boundary
    vertices, whose incident-face fan is one-sided — and their errors are summed
    into a single `MeanAbsoluteError` loss. Differentiating that loss with respect
    to the force densities backpropagates through every vertex normal at once, so
    a `nan` leaking from the `-1`-padding-replacement `jnp.where` on the ragged
    mesh (or from a zero-length normal anywhere) would surface as a non-finite
    gradient entry. The quad grid is the padding-free baseline.
    """
    mesh = request.getfixturevalue(mesh_fixture)

    for vkey in mesh.vertices():
        if mesh.is_vertex_on_boundary(vkey):
            mesh.vertex_support(vkey)
        else:
            mesh.vertex_load(vkey, [0.0, 0.0, -0.1])
    for edge in mesh.edges():
        mesh.edge_forcedensity(edge, -1.0)

    goals = [VertexNormalAngleGoal(vkey, vector=[0.0, 0.0, 1.0], target=TARGET_ANGLE)
             for vkey in mesh.vertices()]
    loss = Loss(MeanAbsoluteError(goals=goals))

    gradient = _loss_grad_wrt_forcedensities(mesh, loss)

    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("goal_cls", [VertexNormalAngleGoal, VertexTangentAngleGoal])
def test_vertex_angle_goal_prediction_shape(ragged_mesh, goal_cls):
    """
    A vertex angle goal's goal and prediction shapes match on a ragged mesh.

    `ragged_mesh` mixes triangular and quad faces, so its `faces_indexed` rows are
    `-1`-padded. This exercises the padding path in the vertex normal without
    tripping the `Goal.__call__` shape assertion.
    """
    mesh, free = _anchored_mesh(ragged_mesh)

    goals = [goal_cls(vkey, vector=[0.0, 0.0, 1.0], target=TARGET_ANGLE)
             for vkey in free]
    loss = Loss(SquaredError(goals=goals))

    # a single optimization step is enough to force the goal to be evaluated
    constrained_fdm(mesh,
                    optimizer=LBFGSB(),
                    loss=loss,
                    parameters=[EdgeForceDensityParameter(edge, BOUND_LOW, BOUND_UP)
                                for edge in mesh.edges()],
                    maxiter=1)
