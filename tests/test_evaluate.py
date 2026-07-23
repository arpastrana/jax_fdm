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
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import DatastructureState
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import datastructure_state
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import model_from_sparsity
from jax_fdm.equilibrium import structure_from_datastructure
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
from jax_fdm.optimization import collect_goals
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


# The maximum iteration count and tolerance the mesh fixtures form-find under, so
# the __call__ reference re-solves with the model settings that produced them.
MESH_TMAX = 10
MESH_ETA = 1e-6

# `evaluate` and `__call__` run the identical deterministic solve, and the
# datastructure round-trips its coordinates through float64 losslessly, so the
# two paths agree to the last bit (measured: exactly zero). Compare with rtol=0
# and a near-exact atol that admits only ULP-scale drift, never solver slack.
ATOL_EXACT = 1e-9


def assert_bit_close(actual, expected):
    """
    Assert two evaluations agree to the last bit, up to ULP-scale drift.
    """
    assert jnp.allclose(actual, expected, rtol=0.0, atol=ATOL_EXACT)


def _face_loaded_mesh():
    """
    A boundary-supported quad meshgrid carrying a vertical area load per face.
    """
    mesh = FDMesh.from_meshgrid(dx=2, nx=4)
    for vertex in mesh.vertices_on_boundary():
        mesh.vertex_support(vertex)
    mesh.edges_forcedensities(-2.0)
    mesh.faces_loads([0.0, 0.0, -0.5])

    return mesh


@pytest.fixture(params=[False, True], ids=["global_load", "local_load"])
def face_loaded_mesh(request):
    """
    A form-found mesh with shape-dependent face area loads, and how it was solved.

    Yields the original mesh, the form-found copy, and the solve settings. The
    fixture is parametrized over ``is_load_local`` so both the global and the
    face-local load paths are exercised.

    A shape-dependent face load only reaches the nodes when ``tmax`` exceeds one,
    so the mesh is form-found iteratively. The original mesh is kept alongside the
    solved one because its parameters are what the iterative solve consumed: the
    form-found copy has already absorbed the distributed face load into its node
    loads, and `evaluate` reads those back as-is without re-aggregating. Solving
    the `__call__` reference from the original parameters is therefore the only
    way to reproduce `evaluate` without double-counting the face load.
    """
    is_load_local = request.param
    original = _face_loaded_mesh()
    form_found = fdm(
        original,
        sparse=False,
        tmax=MESH_TMAX,
        eta=MESH_ETA,
        is_load_local=is_load_local,
    )

    return original, form_found, is_load_local


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
    corrupted table. The vector is an unbatched leaf, so a lone goal carries it
    at shape (3,); evaluation must leave that shape untouched.
    """
    goal = EdgeAngleGoal((1, 2), target=0.0, vector=[0.0, 0.0, 1.0])
    assert goal.vector.shape == (3,)

    first = goal.evaluate(arch, sparse=False)
    assert goal.vector.shape == (3,)

    second = goal.evaluate(arch, sparse=False)
    assert goal.vector.shape == (3,)
    assert jnp.allclose(first.prediction, second.prediction)


@pytest.mark.parametrize(
    "key",
    [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        ((0, 1), (1, 2), (2, 3), (3, 4)),
    ],
    ids=["list", "tuple"],
)
def test_aggregate_goal_evaluate(arch, key):
    """
    An aggregate goal reduces over its whole key sequence in one evaluation.

    Its keys may be a list or a tuple: the aggregate-vs-single distinction is the
    goal's is_aggregate flag, not the key's Python type, so a tuple of edge keys
    must resolve every edge rather than collapse to the first (a single edge key
    is itself a two-int tuple).
    """
    goal = EdgesLengthEqualGoal(key)

    gstate = goal.evaluate(arch, sparse=False)

    # a lone goal returns the raw per-element shape; this scalar aggregate
    # reduces its whole key sequence to a single () value
    assert gstate.prediction.shape == ()
    assert jnp.all(jnp.isfinite(gstate.prediction))


def test_aggregate_goal_rejects_non_sequence_key(recwarn):
    """
    A one-shot iterable key fails at construction, not silently later.

    A generator is not a valid JAX array type, so the key converter's
    `jnp.asarray` rejects it with a `TypeError` right at construction, before
    equinox flattens the goal. The key never reaches `jax.tree_util`, so no
    "treated as a leaf" warning leaks (which a future JAX would turn into an
    error). The converter carries no teaching guard of its own: an off-contract
    key is left to fail on JAX's own terms.
    """
    # A generator is off-contract on purpose: the key type is a sequence, and
    # this guards that the runtime rejects the one-shot iterable cleanly.
    with pytest.raises(TypeError, match="generator"):
        EdgesLengthEqualGoal(
            edge
            for edge in [(0, 1), (1, 2)]  # pyright: ignore[reportArgumentType]
        )

    leaked = [w for w in recwarn if "treated as a leaf" in str(w.message)]
    assert not leaked


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

    # Cross-backend rather than same-solve: dense and sparse assemble the state
    # differently, so this is a genuine float tolerance, not bit-equality. It
    # still lands near the float64 floor because the edge lengths this loss reads
    # come from stored attributes, never the differing connectivity matvec.
    assert jnp.allclose(dense, sparse, rtol=0.0, atol=ATOL_EXACT)


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


# ==============================================================================
# evaluate() equals __call__()
# ==============================================================================
#
# `evaluate` reads the equilibrium state off the datastructure's stored
# attributes; `__call__` consumes (goals, errors) or re-solves for (constraints,
# losses) an equilibrium state. On a form-found datastructure the two paths see
# the same equilibrium, so `evaluate` must reproduce `__call__` field for field.


def _call_ingredients(
    datastructure,
    sparse,
    tmax=1,
    eta=1e-6,
    is_load_local=False,
    params_source=None,
):
    """
    The model, structure, params, and equilibrium state the __call__ paths take.

    Solves with the given model settings so the reference reproduces however the
    datastructure was form-found. `params_source` overrides where the parameters
    are read from; it defaults to `datastructure`, but a form-found mesh must
    pass its pre-solve original so the face load is not distributed twice.
    """
    model = model_from_sparsity(
        sparse=sparse,
        tmax=tmax,
        eta=eta,
        is_load_local=is_load_local,
    )
    structure = structure_from_datastructure(datastructure, sparse)
    params = EquilibriumParametersState.from_datastructure(
        params_source or datastructure,
    )
    eqstate = model(params, structure)

    return model, structure, params, eqstate


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "make_goal",
    [
        lambda: EdgeLengthGoal((1, 2), 2.0),
        lambda: EdgeAngleGoal((1, 2), target=0.0, vector=[0.0, 0.0, 1.0]),
        lambda: NodeResidualForceGoal(2, 0.0),
        lambda: EdgesLengthEqualGoal([(0, 1), (1, 2), (2, 3), (3, 4)]),
        lambda: NetworkLoadPathGoal(),
    ],
    ids=["length", "angle", "residual", "aggregate", "loadpath"],
)
def test_goal_evaluate_matches_call(arch, sparse, make_goal):
    """
    A goal's `evaluate` reproduces `__call__` on the same equilibrium state.

    Covers a scalar, a vector, an aggregate, and a whole-network goal so the
    operand, index, and reshape machinery is exercised on both pipelines.
    """
    goal = make_goal()
    _, structure, _, eqstate = _call_ingredients(arch, sparse)

    called = goal(eqstate, structure)
    evaluated = goal.evaluate(arch, sparse=sparse)

    assert_bit_close(called.goal, evaluated.goal)
    assert_bit_close(called.prediction, evaluated.prediction)
    assert_bit_close(called.weight, evaluated.weight)


@pytest.mark.parametrize("sparse", [False, True])
def test_error_evaluate_matches_call(arch, sparse):
    """
    An error term's `evaluate` reproduces `__call__` on the same equilibrium.

    `__call__` reduces over the goal collections the optimizer batches, while
    `evaluate` reduces over the raw goals; both must land on the same scalar.
    """
    goals = [EdgeLengthGoal(edge, 1.0) for edge in arch.edges()]
    error = SquaredError(goals)
    error.collections = collect_goals(error.goals)

    _, structure, _, eqstate = _call_ingredients(arch, sparse)

    called = error(eqstate, structure)
    evaluated = error.evaluate(arch, sparse=sparse)

    assert_bit_close(called, evaluated)


@pytest.mark.parametrize("sparse", [False, True])
def test_constraint_evaluate_matches_call(arch, sparse):
    """
    A constraint's `evaluate` reproduces `__call__` on the same equilibrium.

    `__call__` solves for equilibrium from raw parameters; `evaluate` reads the
    stored state. On a form-found arch the two agree.
    """
    constraint = EdgeLengthConstraint((2, 3), 0.5, 2.0)
    model, structure, params, _ = _call_ingredients(arch, sparse)

    called = constraint(params, model, structure)
    evaluated = constraint.evaluate(arch, sparse=sparse)

    assert_bit_close(called, evaluated)


@pytest.mark.parametrize("sparse", [False, True])
def test_loss_evaluate_matches_call(arch, sparse):
    """
    A loss's `evaluate` reproduces `__call__` on the same equilibrium.

    Exercises both an error term and a regularizer, so the loss agrees across
    the solving and state-reading paths on both pipelines.
    """
    goals = [EdgeLengthGoal(edge, 1.0) for edge in arch.edges()]
    loss = Loss(SquaredError(goals), L2Regularizer(alpha=0.1))
    for term in loss.terms_error:
        term.collections = collect_goals(term.goals)

    model, structure, params, _ = _call_ingredients(arch, sparse)

    called = loss(params, model, structure)
    evaluated = loss.evaluate(arch, sparse=sparse)

    assert_bit_close(called, evaluated)


# ==============================================================================
# evaluate() equals __call__() with shape-dependent face loads and tmax > 1
# ==============================================================================
#
# When a mesh carries face area loads and is form-found iteratively, the load
# reaching each node depends on the deformed geometry. `evaluate` reads the
# solved node loads straight off the mesh, so it must still match a `__call__`
# that re-runs the same iterative solve from the original parameters. The
# `face_loaded_mesh` fixture spans is_load_local False and True.


def test_face_load_goal_evaluate_matches_call(face_loaded_mesh):
    """
    A goal on a face-loaded, iteratively solved mesh matches its `__call__`.
    """
    original, form_found, is_load_local = face_loaded_mesh
    goal = EdgeLengthGoal(list(form_found.edges())[5], 1.0)

    _, structure, _, eqstate = _call_ingredients(
        form_found,
        sparse=False,
        tmax=MESH_TMAX,
        eta=MESH_ETA,
        is_load_local=is_load_local,
        params_source=original,
    )

    called = goal(eqstate, structure)
    evaluated = goal.evaluate(form_found, sparse=False)

    assert_bit_close(called.prediction, evaluated.prediction)


def test_face_load_error_evaluate_matches_call(face_loaded_mesh):
    """
    An error term on a face-loaded, iteratively solved mesh matches its `__call__`.
    """
    original, form_found, is_load_local = face_loaded_mesh
    goals = [EdgeLengthGoal(edge, 1.0) for edge in form_found.edges()]
    error = SquaredError(goals)
    error.collections = collect_goals(error.goals)

    _, structure, _, eqstate = _call_ingredients(
        form_found,
        sparse=False,
        tmax=MESH_TMAX,
        eta=MESH_ETA,
        is_load_local=is_load_local,
        params_source=original,
    )

    called = error(eqstate, structure)
    evaluated = error.evaluate(form_found, sparse=False)

    assert_bit_close(called, evaluated)


def test_face_load_constraint_evaluate_matches_call(face_loaded_mesh):
    """
    A constraint on a face-loaded, iteratively solved mesh matches its `__call__`.
    """
    original, form_found, is_load_local = face_loaded_mesh
    constraint = EdgeLengthConstraint(list(form_found.edges())[5], 0.1, 5.0)

    model, structure, params, _ = _call_ingredients(
        form_found,
        sparse=False,
        tmax=MESH_TMAX,
        eta=MESH_ETA,
        is_load_local=is_load_local,
        params_source=original,
    )

    called = constraint(params, model, structure)
    evaluated = constraint.evaluate(form_found, sparse=False)

    assert_bit_close(called, evaluated)


def test_face_load_loss_evaluate_matches_call(face_loaded_mesh):
    """
    A loss on a face-loaded, iteratively solved mesh matches its `__call__`.
    """
    original, form_found, is_load_local = face_loaded_mesh
    goals = [EdgeLengthGoal(edge, 1.0) for edge in form_found.edges()]
    loss = Loss(SquaredError(goals), L2Regularizer(alpha=0.1))
    for term in loss.terms_error:
        term.collections = collect_goals(term.goals)

    model, structure, params, _ = _call_ingredients(
        form_found,
        sparse=False,
        tmax=MESH_TMAX,
        eta=MESH_ETA,
        is_load_local=is_load_local,
        params_source=original,
    )

    called = loss(params, model, structure)
    evaluated = loss.evaluate(form_found, sparse=False)

    assert_bit_close(called, evaluated)


def test_face_load_double_count_guard(face_loaded_mesh):
    """
    Re-reading parameters off the form-found mesh double-counts the face load.

    Pins the fixture's reason for keeping the original mesh: solving from the
    form-found mesh's own parameters redistributes the face load a second time,
    so `evaluate` and that mistaken `__call__` must visibly disagree. If this
    ever stops disagreeing, the fixture no longer tests the tmax > 1 face-load
    path and the other assertions here go slack.
    """
    _, form_found, is_load_local = face_loaded_mesh
    goals = [EdgeLengthGoal(edge, 1.0) for edge in form_found.edges()]
    loss = Loss(SquaredError(goals), L2Regularizer(alpha=0.1))
    for term in loss.terms_error:
        term.collections = collect_goals(term.goals)

    # params_source defaults to the form-found mesh: the wrong reference.
    model, structure, params, _ = _call_ingredients(
        form_found,
        sparse=False,
        tmax=MESH_TMAX,
        eta=MESH_ETA,
        is_load_local=is_load_local,
    )

    called_double = loss(params, model, structure)
    evaluated = loss.evaluate(form_found, sparse=False)

    # The gap is the doubly-distributed face load, not numerical noise: assert it
    # is a substantial fraction of the loss, so a shrinking face load can never
    # let this pass by accident.
    gap = jnp.abs(called_double - evaluated)
    assert gap > 1e-3 * jnp.abs(evaluated)
