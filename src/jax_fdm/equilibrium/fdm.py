import numpy as np

from jax_fdm import DTYPE_JAX

from jax_fdm.datastructures import FDNetwork
from jax_fdm.datastructures import FDMesh

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumModelSparse

from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructureSparse
from jax_fdm.equilibrium import EquilibriumMeshStructureSparse

from jax_fdm.equilibrium import EquilibriumParametersState


# ==========================================================================
# Form-finding
# ==========================================================================

def _fdm(model, params, structure, datastructure):
    """
    Compute a datastructure in a state of static equilibrium using the force density method.
    """
    # compute static equilibrium
    eq_state = model(params, structure)

    # update equilibrium state in a copy of the datastructure
    return datastructure_updated(datastructure, eq_state, params)


def fdm(datastructure,
        sparse=True,
        is_load_local=False,
        tmax=1,
        eta=1e-6,
        itersolve_fn=None,
        implicit_diff=True,
        verbose=False):
    """
    Compute a datastructure in a state of static equilibrium using the force density method.
    """
    datastructure_validate(datastructure)

    model = model_from_sparsity(sparse, tmax, eta, is_load_local, itersolve_fn, implicit_diff, verbose)
    structure = structure_from_datastructure(datastructure, sparse)

    params = EquilibriumParametersState.from_datastructure(datastructure, dtype=DTYPE_JAX)

    return _fdm(model, params, structure, datastructure)


# ==========================================================================
# Constrained form-finding
# ==========================================================================

def constrained_fdm(datastructure,
                    optimizer,
                    loss,
                    parameters=None,
                    constraints=None,
                    maxiter=100,
                    tol=1e-6,
                    tmax=1,
                    eta=1e-6,
                    callback=None,
                    sparse=True,
                    is_load_local=False,
                    itersolve_fn=None,
                    implicit_diff=True,
                    verbose=False,
                    jit=True):
    """
    Generate a network in a constrained state of static equilibrium using the force density method.
    """
    datastructure_validate(datastructure)

    if constraints and sparse:
        print("\nConstraints are not supported yet for sparse inputs. Switching to dense.")
        sparse = False

    model = model_from_sparsity(sparse, tmax, eta, is_load_local, itersolve_fn, implicit_diff, verbose)
    structure = structure_from_datastructure(datastructure, sparse)

    opt_problem = optimizer.problem(model,
                                    structure,
                                    datastructure,
                                    loss,
                                    parameters,
                                    constraints,
                                    maxiter,
                                    tol,
                                    callback,
                                    jit)

    opt_params = optimizer.solve(opt_problem)

    params = optimizer.parameters_fdm(opt_params)

    return _fdm(model, params, structure, datastructure)


# ==========================================================================
# Helpers
# ==========================================================================

def model_from_sparsity(sparse, tmax, eta, is_load_local, itersolve_fn=None, implicit_diff=True, verbose=False):
    """
    Create an equilibrium model from a sparsity flag.
    """
    model = EquilibriumModel
    if sparse:
        model = EquilibriumModelSparse

    return model(tmax, eta, is_load_local, itersolve_fn, implicit_diff, verbose=verbose)


def structure_from_datastructure(datastructure, sparse):
    """
    Create a structure from a force density datastructure.
    """
    if isinstance(datastructure, FDNetwork):
        structure_factory = structure_from_network
    elif isinstance(datastructure, FDMesh):
        structure_factory = structure_from_mesh
    else:
        raise ValueError(f"Input datastructure {datastructure} is invalid")

    return structure_factory(datastructure, sparse)


def structure_from_network(network, sparse):
    """
    Create a structure from a network.
    """
    structure = EquilibriumStructure
    if sparse:
        structure = EquilibriumStructureSparse

    return structure.from_network(network)


def structure_from_mesh(mesh, sparse):
    """
    Create a structure from a mesh.
    """
    structure = EquilibriumMeshStructure
    if sparse:
        structure = EquilibriumMeshStructureSparse

    return structure.from_mesh(mesh)


def datastructure_validate(datastructure):
    """
    Check that the network is healthy.
    """
    assert datastructure.number_of_supports() > 0, "The FD datastructure has no supports"
    assert datastructure.number_of_edges() > 0, "The FD datastructure has no edges"

    has_fd = np.abs(np.array(datastructure.edges_forcedensities())) > 0.0
    num_no_fd = np.sum(np.logical_not(has_fd).astype(float))
    assert np.all(has_fd), f"The FD datastructure has {int(num_no_fd)} edges with zero force density"

    try:
        assert datastructure.number_of_nodes() > 0, "The FD datastructure has no nodes"
    except AttributeError:
        assert datastructure.number_of_vertices() > 0, "The FD datastructure has no nodes"


def datastructure_updated(datastructure, eq_state, params, use_loadsfromparams=False):
    """
    Return a copy of a datastructure whose attributes are updated with an equilibrium state.
    """
    datastructure = datastructure.copy()
    datastructure_update(datastructure, eq_state, params, use_loadsfromparams)

    return datastructure


def datastructure_update(datastructure, eq_state, params, use_loadsfromparams=False):
    """
    Update in-place the attributes of a datastructure with an equilibrium state.
    """
    # unpack equilibrium state
    xyz = eq_state.xyz.tolist()
    lengths = eq_state.lengths.tolist()
    residuals = eq_state.residuals.tolist()
    forces = eq_state.forces.tolist()

    forcedensities = params.q.tolist()

    if use_loadsfromparams:
        loads = params.loads.nodes.tolist()
    else:
        loads = eq_state.loads.tolist()

    # update edges
    datastructure_edges_update(datastructure,
                               (lengths, forces, forcedensities))

    # update nodes / vertices
    datastructure_nodes_update(datastructure,
                               (xyz, residuals, loads))


def datastructure_edges_update(datastructure, eqstate_edges):
    """
    Update the edge attributes of a datastructure.
    """
    lengths, forces, forcedensities = eqstate_edges

    for idx, edge in datastructure.index_uv().items():
        datastructure.edge_attribute(edge, name="length", value=lengths[idx].pop())
        datastructure.edge_attribute(edge, name="force", value=forces[idx].pop())
        datastructure.edge_attribute(edge, name="q", value=forcedensities[idx])


def datastructure_nodes_update(datastructure, eqstate_nodes):
    """
    Update the nodes or vertex attributes of a datastructure.
    """
    xyz, residuals, loads = eqstate_nodes

    if isinstance(datastructure, FDNetwork):
        nodevertex_updater = datastructure.node_attribute
    elif isinstance(datastructure, FDMesh):
        nodevertex_updater = datastructure.vertex_attribute
    else:
        raise ValueError(f"Input datastructure {datastructure} is invalid")

    for idx, key in datastructure.index_key().items():

        for name, value in zip("xyz", xyz[idx]):
            nodevertex_updater(key, name=name, value=value)

        for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
            nodevertex_updater(key, name=name, value=value)

        for name, value in zip(["px", "py", "pz"], loads[idx]):
            nodevertex_updater(key, name=name, value=value)
