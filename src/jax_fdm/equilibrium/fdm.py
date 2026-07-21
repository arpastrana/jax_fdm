from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium.models import EquilibriumModel
from jax_fdm.equilibrium.models import EquilibriumModelSparse
from jax_fdm.equilibrium.states import EquilibriumParametersState
from jax_fdm.equilibrium.states import EquilibriumState
from jax_fdm.equilibrium.structures import EquilibriumMeshStructure
from jax_fdm.equilibrium.structures import EquilibriumMeshStructureSparse
from jax_fdm.equilibrium.structures import EquilibriumStructure
from jax_fdm.equilibrium.structures import EquilibriumStructureSparse

# Imported for annotations only: a runtime import would close the package cycle
# equilibrium -> losses/optimization/parameters -> goals -> equilibrium and make
# `import jax_fdm.goals` order-dependent.
if TYPE_CHECKING:
    from jax_fdm.constraints import Constraint
    from jax_fdm.losses import Loss
    from jax_fdm.optimization import Optimizer
    from jax_fdm.parameters import Parameter

__all__ = [
    "fdm",
    "constrained_fdm",
    "model_from_sparsity",
    "structure_from_datastructure",
    "structure_from_network",
    "structure_from_mesh",
    "datastructure_validate",
    "datastructure_updated",
    "datastructure_update",
    "datastructure_edges_update",
    "datastructure_nodes_update",
]

# ==========================================================================
# Type aliases
# ==========================================================================

# Per-element equilibrium state columns, as returned by Array.tolist(): a
# scalar value per element (e.g. force densities) or an xyz triple per element.
ElementScalars = list[float]
ElementVectors = list[list[float]]

# Binds a function's return type to its input type: form-finding a network
# yields a network, a mesh yields a mesh — the result is an updated copy of
# the input.
AnyFDDatastructure = TypeVar("AnyFDDatastructure", bound=FDNetwork | FDMesh)

# ==========================================================================
# Form-finding
# ==========================================================================


def _fdm(
    model: EquilibriumModel,
    params: EquilibriumParametersState,
    structure: EquilibriumStructure,
    datastructure: AnyFDDatastructure,
) -> AnyFDDatastructure:
    """
    Solve for static equilibrium and write the result into a datastructure copy.

    Parameters
    ----------
    model :
        The equilibrium model that computes the equilibrium state.
    params :
        The force densities, fixed node coordinates, and load state.
    structure :
        The structure that provides the connectivity matrices.
    datastructure :
        The network or mesh to update with the equilibrium state.

    Returns
    -------
    datastructure :
        A copy of the input datastructure updated with the equilibrium state.
    """
    # compute static equilibrium
    eq_state = model(params, structure)

    # update equilibrium state in a copy of the datastructure
    return datastructure_updated(datastructure, eq_state, params)


def fdm(
    datastructure: AnyFDDatastructure,
    sparse: bool = True,
    is_load_local: bool = False,
    tmax: int = 1,
    eta: float = 1e-6,
    itersolve_fn: Callable | None = None,
    iterload_fn: Callable | None = None,
    implicit_diff: bool = True,
    verbose: bool = False,
) -> AnyFDDatastructure:
    """
    Compute a datastructure in static equilibrium with the force density method.

    Parameters
    ----------
    datastructure :
        The network or mesh to form-find.
    sparse :
        If True, assemble and solve the equilibrium system with a sparse solver.
    is_load_local :
        If True, apply edge and face loads in their local coordinate systems
        (follower loads).
    tmax :
        The maximum number of iterations. With ``tmax=1`` a single linear FDM step
        is taken and edge and face loads are ignored.
    eta :
        The convergence tolerance for the iterative solve.
    itersolve_fn :
        The iterative equilibrium solver. If None, forward fixed-point iteration
        is used.
    iterload_fn :
        A load callback invoked once before iterative equilibrium starts.
    implicit_diff :
        If True, apply implicit differentiation for the backward pass.
    verbose :
        Whether to print calculation info to the terminal.

    Returns
    -------
    datastructure :
        A copy of the input datastructure in static equilibrium.
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
        verbose=verbose,
    )
    structure = structure_from_datastructure(datastructure, sparse)

    params = EquilibriumParametersState.from_datastructure(
        datastructure,
        dtype=DTYPE_JAX,
    )

    return _fdm(model, params, structure, datastructure)


# ==========================================================================
# Constrained form-finding
# ==========================================================================


def constrained_fdm(
    datastructure: AnyFDDatastructure,
    optimizer: "Optimizer",
    loss: "Loss",
    parameters: Sequence["Parameter"] | None = None,
    constraints: Sequence["Constraint"] | None = None,
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
    jit: bool = True,
) -> AnyFDDatastructure:
    """
    Form-find a datastructure in constrained static equilibrium via optimization.

    Parameters
    ----------
    datastructure :
        The network or mesh to form-find.
    optimizer :
        The optimizer that minimizes the loss over the design parameters.
    loss :
        The loss function assembled from goals to minimize.
    parameters :
        The optimization parameters. If None, the optimizer uses its defaults.
    constraints :
        The constraints to enforce during optimization. If None, the problem is
        unconstrained.
    maxiter :
        The maximum number of optimization iterations.
    tol :
        The convergence tolerance of the optimizer.
    tmax :
        The maximum number of equilibrium iterations per optimization step. With
        ``tmax=1`` a single linear FDM step is taken.
    eta :
        The convergence tolerance for the iterative equilibrium solve.
    callback :
        A function invoked once per optimization iteration.
    sparse :
        If True, assemble and solve the equilibrium system with a sparse solver.
        Forced to False when constraints are given, which sparse does not support.
    is_load_local :
        If True, apply edge and face loads in their local coordinate systems
        (follower loads).
    itersolve_fn :
        The iterative equilibrium solver. If None, forward fixed-point iteration
        is used.
    iterload_fn :
        A load callback invoked once before iterative equilibrium starts.
    implicit_diff :
        If True, apply implicit differentiation for the backward pass.
    nd :
        Unused; kept for backward compatibility.
    verbose :
        Whether to print calculation info to the terminal.
    jit :
        If True, just-in-time compile the optimization problem.

    Returns
    -------
    datastructure :
        A copy of the input datastructure in constrained static equilibrium.

    Notes
    -----
    Constraints are not yet supported for sparse inputs; passing constraints with
    ``sparse=True`` switches the solve to dense and prints a warning.
    """
    datastructure_validate(datastructure)

    if constraints and sparse:
        print(
            "\nConstraints are not supported yet for sparse inputs. "
            "Switching to dense.",
        )
        sparse = False

    model = model_from_sparsity(
        sparse=sparse,
        tmax=tmax,
        eta=eta,
        is_load_local=is_load_local,
        itersolve_fn=itersolve_fn,
        iterload_fn=iterload_fn,
        implicit_diff=implicit_diff,
        verbose=verbose,
    )

    structure = structure_from_datastructure(datastructure, sparse)

    opt_problem = optimizer.problem(
        model,
        structure,
        datastructure,
        loss,
        parameters,
        constraints,
        maxiter,
        tol,
        callback,
        jit,
    )

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
    verbose: bool = False,
) -> EquilibriumModel:
    """
    Instantiate a dense or sparse equilibrium model from a sparsity flag.

    Parameters
    ----------
    sparse :
        If True, instantiate the sparse model; otherwise the dense model.
    tmax :
        The maximum number of equilibrium iterations.
    eta :
        The convergence tolerance for the iterative solve.
    is_load_local :
        If True, apply edge and face loads in their local coordinate systems.
    itersolve_fn :
        The iterative equilibrium solver. If None, the model default is used.
    iterload_fn :
        A load callback invoked once before iterative equilibrium starts.
    implicit_diff :
        If True, apply implicit differentiation for the backward pass.
    verbose :
        Whether to print calculation info to the terminal.

    Returns
    -------
    model :
        The configured equilibrium model.
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
        verbose=verbose,
    )

    return model_instance


def structure_from_datastructure(
    datastructure: FDNetwork | FDMesh,
    sparse: bool,
) -> EquilibriumStructure:
    """
    Build the equilibrium structure matching a network or mesh.

    Parameters
    ----------
    datastructure :
        The network or mesh to derive connectivity from.
    sparse :
        If True, build the sparse structure variant.

    Returns
    -------
    structure :
        The structure that carries the connectivity matrices.

    Raises
    ------
    ValueError
        If the datastructure is neither a network nor a mesh.
    """
    # Call each factory inside its own isinstance branch so the narrowed
    # datastructure type flows into the matching factory signature.
    if isinstance(datastructure, FDNetwork):
        return structure_from_network(datastructure, sparse)
    elif isinstance(datastructure, FDMesh):
        return structure_from_mesh(datastructure, sparse)
    else:
        raise ValueError(f"Input datastructure {datastructure} is invalid")


def structure_from_network(network: FDNetwork, sparse: bool) -> EquilibriumStructure:
    """
    Build the equilibrium structure of a network.

    Parameters
    ----------
    network :
        The network to derive connectivity from.
    sparse :
        If True, build the sparse structure variant.

    Returns
    -------
    structure :
        The structure that carries the network connectivity matrices.
    """
    structure: type[EquilibriumStructure] = EquilibriumStructure
    if sparse:
        structure = EquilibriumStructureSparse

    return structure.from_network(network)


def structure_from_mesh(mesh: FDMesh, sparse: bool) -> EquilibriumMeshStructure:
    """
    Build the equilibrium structure of a mesh.

    Parameters
    ----------
    mesh :
        The mesh to derive connectivity from.
    sparse :
        If True, build the sparse structure variant.

    Returns
    -------
    structure :
        The mesh structure that carries connectivity and face topology.
    """
    structure: type[EquilibriumMeshStructure] = EquilibriumMeshStructure
    if sparse:
        structure = EquilibriumMeshStructureSparse

    return structure.from_mesh(mesh)


def datastructure_validate(datastructure: FDNetwork | FDMesh) -> None:
    """
    Assert that a datastructure is well-posed for form-finding.

    Parameters
    ----------
    datastructure :
        The network or mesh to validate.

    Raises
    ------
    AssertionError
        If the datastructure has no supports, no edges, any edge with zero force
        density, or no nodes (network) or vertices (mesh).
    """
    assert datastructure.number_of_supports() > 0, (
        "The FD datastructure has no supports"
    )
    assert datastructure.number_of_edges() > 0, "The FD datastructure has no edges"

    has_fd = np.abs(np.array(datastructure.edges_forcedensities())) > 0.0
    num_no_fd = np.sum(np.logical_not(has_fd).astype(float))
    assert np.all(has_fd), (
        f"The FD datastructure has {int(num_no_fd)} edges with zero force density"
    )

    if isinstance(datastructure, FDNetwork):
        assert datastructure.number_of_nodes() > 0, "The FD datastructure has no nodes"
    elif isinstance(datastructure, FDMesh):
        assert datastructure.number_of_vertices() > 0, (
            "The FD datastructure has no vertices"
        )


def datastructure_updated(
    datastructure: AnyFDDatastructure,
    eq_state: EquilibriumState,
    params: EquilibriumParametersState,
    use_loadsfromparams: bool = False,
) -> AnyFDDatastructure:
    """
    Return a copy of a datastructure updated with an equilibrium state.

    Parameters
    ----------
    datastructure :
        The network or mesh to update.
    eq_state :
        The equilibrium state to write into the datastructure.
    params :
        The parameter state, used as the load source when requested.
    use_loadsfromparams :
        If True, take node loads from ``params`` instead of the equilibrium state.

    Returns
    -------
    datastructure :
        A copy of the input datastructure with updated node and edge attributes.
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
    Update the attributes of a datastructure in place with an equilibrium state.

    Parameters
    ----------
    datastructure :
        The network or mesh to update in place.
    eq_state :
        The equilibrium state to write into the datastructure.
    params :
        The parameter state, used as the load source when requested.
    use_loadsfromparams :
        If True, take node loads from ``params`` instead of the equilibrium state.
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
    datastructure_edges_update(datastructure, (lengths, forces, forcedensities))

    # update nodes / vertices
    datastructure_nodes_update(datastructure, (xyz, residuals, loads))


def datastructure_edges_update(
    datastructure: FDNetwork | FDMesh,
    eqstate_edges: tuple[ElementVectors, ElementVectors, ElementScalars],
) -> None:
    """
    Write per-edge length, force, and force density onto a datastructure.

    Parameters
    ----------
    datastructure :
        The network or mesh whose edge attributes are updated in place.
    eqstate_edges :
        The per-edge lengths, forces, and force densities, in that order.
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
    Write per-node coordinates, residuals, and loads onto a datastructure.

    Parameters
    ----------
    datastructure :
        The network or mesh whose node or vertex attributes are updated in place.
    eqstate_nodes :
        The per-node coordinates, residuals, and loads, in that order.

    Raises
    ------
    ValueError
        If the datastructure is neither a network nor a mesh.
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
