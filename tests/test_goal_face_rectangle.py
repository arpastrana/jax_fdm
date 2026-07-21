"""
Characterization tests for the face rectangularity goal.

The goal once cached only its own faces' corner indices at init and summed
over every cached row in each vmapped prediction call, so two goals collected
into one vectorized collection each received the sum over both faces. The
collection parity test pins the per-index gather that fixed this.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.goals import FaceRectangularGoal
from jax_fdm.optimization.collections import collect_goals


@pytest.fixture
def skewed_mesh_state(meshgrid_mesh):
    """
    A fully anchored meshgrid with one vertex nudged to skew its faces.

    Every vertex is a support, so the equilibrium state reproduces the input
    coordinates and predictions reflect the authored geometry. The nudge makes
    the per-face rectangularity measures differ, so a summed prediction cannot
    masquerade as a per-face one.
    """
    mesh = meshgrid_mesh
    vkey = next(
        vkey for vkey in mesh.vertices() if not mesh.is_vertex_on_boundary(vkey)
    )
    x, y, z = mesh.vertex_coordinates(vkey)
    mesh.vertex_attributes(vkey, "xyz", [x + 0.3, y + 0.15, z])

    for vkey in mesh.vertices():
        mesh.vertex_support(vkey)
    for edge in mesh.edges():
        mesh.edge_forcedensity(edge, -1.0)

    structure = EquilibriumMeshStructure.from_mesh(mesh)
    model = EquilibriumModel(tmax=1)
    parameters = EquilibriumParametersState.from_datastructure(mesh, dtype=DTYPE_JAX)
    eqstate = model(parameters, structure)

    return mesh, model, structure, eqstate


def test_collected_rectangle_goals_match_individual_predictions(skewed_mesh_state):
    """
    A two-goal collection predicts each face's own rectangularity measure.
    """
    mesh, model, structure, eqstate = skewed_mesh_state
    fkeys = list(mesh.faces())[:2]

    goals = [FaceRectangularGoal(fkey) for fkey in fkeys]
    (collection,) = collect_goals(goals)

    goal_state = collection(eqstate, structure)

    predictions_individual = []
    for fkey in fkeys:
        goal = FaceRectangularGoal(fkey)
        predictions_individual.append(goal(eqstate, structure).prediction)
    predictions_individual = jnp.concatenate(predictions_individual)

    assert goal_state.prediction.shape == predictions_individual.shape
    assert jnp.allclose(goal_state.prediction, predictions_individual)
    # the skew guarantees the two faces measure differently, so a summed
    # prediction leaking across elements would fail the parity above
    assert not jnp.allclose(predictions_individual[0], predictions_individual[1])


def test_collected_rectangle_goal_gradient_stays_per_face(skewed_mesh_state):
    """
    Each collected element's gradient touches only its own face's vertices.
    """
    mesh, model, structure, eqstate = skewed_mesh_state
    fkeys = list(mesh.faces())[:2]

    (collection,) = collect_goals([FaceRectangularGoal(fkey) for fkey in fkeys])

    for index in collection.indices(structure).ravel():

        def prediction_of_xyz(xyz, index=int(index)):
            eq_state = eqstate._replace(xyz=xyz)
            return collection.prediction(eq_state, structure, jnp.asarray(index))

        grad_xyz = jax.jit(jax.grad(prediction_of_xyz))(eqstate.xyz)
        rows_hit = {int(r) for r in jnp.flatnonzero(jnp.any(grad_xyz != 0.0, axis=1))}
        face_rows = {int(v) for v in structure.faces_indexed[int(index), :4]}
        assert rows_hit <= face_rows
