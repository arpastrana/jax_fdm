from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumMeshStructureSparse
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumModelSparse
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import EquilibriumStructureSparse
from jax_fdm.losses import Loss
from jax_fdm.optimization import Optimizer
from jax_fdm.parameters import Parameter

if TYPE_CHECKING:
    from jax_fdm.constraints import Constraint

# ==========================================================================
# Type aliases
# ==========================================================================

# Per-element equilibrium state columns, as returned by Array.tolist(): a
# scalar value per element (e.g. force densities) or an xyz triple per element.
ElementScalars = list[float]
ElementVectors = list[list[float]]

# ==========================================================================
# Form-finding
# ==========================================================================

def _fdm(
    model: EquilibriumModel,
    params: EquilibriumParametersState,
    structure: EquilibriumStructure,
    datastructure: FDNetwork | FDMesh,
) -> FDNetwork | FDMesh:
    """
    Compute a datastructure in a state of static equilibrium using the force density method.
    """
    # compute static equilibrium
    eq_state = model(params, structure)

    # update equilibrium state in a copy of the datastructure
    return datastructure_updated(datastructure, eq_state, params)


def fdm(datastructure: FDNetwork | FDMesh,
        sparse: bool = True,
        is_load_local: bool = False,
        tmax: int = 1,
        eta: float = 1e-6,
        itersolve_fn: Callable | None = None,
        iterload_fn: Callable | None = None,
        implicit_diff: bool = True,
        verbose: bool = False) -> FDNetwork | FDMesh:
    """
    Compute a datastructure in a state of static equilibrium using the force density method.
    """
    datastructure_validate(datastructure)

    model = model_from_sparsity(
        sparse=sparse,
        tmax=tmax,
        eta=eta,
        is_load_local=is_load_local,
        itersolve_fn=itersolve_fn,
        iterload_fn=iterload_fn,
        implicit_diff=implicit_diff,
        verbose=verbose
    )
    structure = structure_from_datastructure(datastructure, sparse)

    params = EquilibriumParametersState.from_datastructure(datastructure, dtype=DTYPE_JAX)

    return _fdm(model, params, structure, datastructure)


# ==========================================================================
# Constrained form-finding
# ==========================================================================

def constrained_fdm(datastructure: FDNetwork | FDMesh,
                    optimizer: Optimizer,
                    loss: Loss,
                    parameters: list[Parameter] | None = None,
                    constraints: list["Constraint"] | None = None,
                    maxiter: int = 100,
                    tol: float = 1e-6,
                    tmax: int = 1,
                    eta: float = 1e-6,
                    callback: Callable | None = None,
                    sparse: bool = True,
                    is_load_local: bool = False,
                    itersolve_fn: Callable | None = None,
                    iterload_fn: Callable | None = None,
                    implicit_diff: bool = True,
                    nd: bool = False,
                    verbose: bool = False,
                    jit: bool = True) -> FDNetwork | FDMesh:
    """
    Generate a network in a constrained state of static equilibrium using the force density method.
    """
    datastructure_validate(datastructure)

    if constraints and sparse:
        print("\nConstraints are not supported yet for sparse inputs. Switching to dense.")
        sparse = False

    model = model_from_sparsity(
        sparse=sparse,
        tmax=tmax,
        eta=eta,
        is_load_local=is_load_local,
        itersolve_fn=itersolve_fn,
        iterload_fn=iterload_fn,
        implicit_diff=implicit_diff,
        verbose=verbose
    )

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

def model_from_sparsity(
        sparse: bool,
        tmax: int,
        eta: float,
        is_load_local: bool = False,
        itersolve_fn: Callable | None = None,
        iterload_fn: Callable | None = None,
        implicit_diff: bool = True,
        verbose: bool = False) -> EquilibriumModel:
    """
    Create an equilibrium model from a sparsity flag.
    """
    model: type[EquilibriumModel] = EquilibriumModel
    if sparse:
        model = EquilibriumModelSparse

    model_instance = model(
        tmax=tmax,
        eta=eta,
        is_load_local=is_load_local,
        itersolve_fn=itersolve_fn,
        iterload_fn=iterload_fn,
        implicit_diff=implicit_diff,
        verbose=verbose)

    return model_instance


def structure_from_datastructure(datastructure: FDNetwork | FDMesh, sparse: bool) -> EquilibriumStructure:
    """
    Create a structure from a force density datastructure.
    """
    if isinstance(datastructure, FDNetwork):
        structure_factory = structure_from_network
    elif isinstance(datastructure, FDMesh):
        structure_factory = structure_from_mesh
    else:
        raise ValueError(f"Input datastructure {datastructure} is invalid")

    return structure_factory(datastructure, sparse)  # pyright: ignore[reportArgumentType]  # structure_factory is narrowed to structure_from_network/structure_from_mesh by the isinstance checks above, but pyright does not carry that correlation across the reassigned datastructure param; the two branches match datastructure's actual runtime type


def structure_from_network(network: FDNetwork, sparse: bool) -> EquilibriumStructure:
    """
    Create a structure from a network.
    """
    structure: type[EquilibriumStructure] = EquilibriumStructure
    if sparse:
        structure = EquilibriumStructureSparse

    return structure.from_network(network)


def structure_from_mesh(mesh: FDMesh, sparse: bool) -> EquilibriumMeshStructure:
    """
    Create a structure from a mesh.
    """
    structure: type[EquilibriumMeshStructure] = EquilibriumMeshStructure
    if sparse:
        structure = EquilibriumMeshStructureSparse

    return structure.from_mesh(mesh)


def datastructure_validate(datastructure: FDNetwork | FDMesh) -> None:
    """
    Check that the network is healthy.
    """
    assert datastructure.number_of_supports() > 0, "The FD datastructure has no supports"
    assert datastructure.number_of_edges() > 0, "The FD datastructure has no edges"

    has_fd = np.abs(np.array(datastructure.edges_forcedensities())) > 0.0
    num_no_fd = np.sum(np.logical_not(has_fd).astype(float))
    assert np.all(has_fd), f"The FD datastructure has {int(num_no_fd)} edges with zero force density"

    if isinstance(datastructure, FDNetwork):
        assert datastructure.number_of_nodes() > 0, "The FD datastructure has no nodes"
    elif isinstance(datastructure, FDMesh):
        assert datastructure.number_of_vertices() > 0, "The FD datastructure has no vertices"


def datastructure_updated(
    datastructure: FDNetwork | FDMesh,
    eq_state: EquilibriumState,
    params: EquilibriumParametersState,
    use_loadsfromparams: bool = False,
) -> FDNetwork | FDMesh:
    """
    Return a copy of a datastructure whose attributes are updated with an equilibrium state.
    """
    datastructure = datastructure.copy()
    datastructure_update(datastructure, eq_state, params, use_loadsfromparams)

    return datastructure


def datastructure_update(
    datastructure: FDNetwork | FDMesh,
    eq_state: EquilibriumState,
    params: EquilibriumParametersState,
    use_loadsfromparams: bool = False,
) -> None:
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


def datastructure_edges_update(
    datastructure: FDNetwork | FDMesh,
    eqstate_edges: tuple[ElementVectors, ElementVectors, ElementScalars],
) -> None:
    """
    Update the edge attributes of a datastructure.
    """
    lengths, forces, forcedensities = eqstate_edges

    for idx, edge in datastructure.index_edge().items():
        datastructure.edge_attribute(edge, name="length", value=lengths[idx].pop())
        datastructure.edge_attribute(edge, name="force", value=forces[idx].pop())
        datastructure.edge_attribute(edge, name="q", value=forcedensities[idx])


def datastructure_nodes_update(
    datastructure: FDNetwork | FDMesh,
    eqstate_nodes: tuple[ElementVectors, ElementVectors, ElementVectors],
) -> None:
    """
    Update the nodes or vertex attributes of a datastructure.
    """
    xyz, residuals, loads = eqstate_nodes

    if isinstance(datastructure, FDNetwork):
        nodevertex_updater = datastructure.node_attribute
        index_key = datastructure.index_node
    elif isinstance(datastructure, FDMesh):
        nodevertex_updater = datastructure.vertex_attribute
        index_key = datastructure.index_vertex
    else:
        raise ValueError(f"Input datastructure {datastructure} is invalid")

    for idx, key in index_key().items():

        for name, value in zip("xyz", xyz[idx]):
            nodevertex_updater(key, name=name, value=value)

        for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
            nodevertex_updater(key, name=name, value=value)

        for name, value in zip(["px", "py", "pz"], loads[idx]):
            nodevertex_updater(key, name=name, value=value)
