import numpy as np

from jax_fdm import DTYPE_NP

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumModelSparse

from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import EquilibriumStructureSparse


# ==========================================================================
# Form-finding
# ==========================================================================

def _fdm(model, params, structure, network):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    # compute static equilibrium
    eq_state = model(params, structure)

    # update equilibrium state in a copy of the network
    return network_updated(network, eq_state)


def fdm(network, sparse=True):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    network_validate(network)

    model = model_from_network(network, sparse)
    structure = structure_from_network(network, sparse)

    params = [np.array(p, dtype=DTYPE_NP) for p in network.parameters()]

    return _fdm(model, params, structure, network)


# ==========================================================================
# Constrained form-finding
# ==========================================================================

def constrained_fdm(network,
                    optimizer,
                    loss,
                    parameters=None,
                    constraints=None,
                    maxiter=100,
                    tol=1e-6,
                    callback=None,
                    sparse=True):
    """
    Generate a network in a constrained state of static equilibrium using the force density method.
    """
    network_validate(network)

    if constraints and sparse:
        print("Constraints are not supported yet for sparse inputs. Switching to dense.")
        sparse = False

    model = model_from_network(network, sparse)
    structure = structure_from_network(network, sparse)

    opt_problem = optimizer.problem(model, structure, loss, parameters, constraints, maxiter, tol, callback)
    opt_params = optimizer.solve(opt_problem)
    params = optimizer.parameters_fdm(opt_params)

    return _fdm(model, params, structure, network)


# ==========================================================================
# Helpers
# ==========================================================================

def model_from_network(network, sparse):
    """
    Create an equilibrium model from a network.
    """
    model = EquilibriumModel
    if sparse:
        model = EquilibriumModelSparse

    return model.from_network(network)


def structure_from_network(network, sparse):
    """
    Create a structure from a network.
    """
    structure = EquilibriumStructure
    if sparse:
        structure = EquilibriumStructureSparse

    return structure.from_network(network)


def network_validate(network):
    """
    Check that the network is healthy.
    """
    assert network.number_of_supports() > 0, "The network has no supports"
    assert network.number_of_edges() > 0, "The network has no edges"
    assert network.number_of_nodes() > 0, "The network has no nodes"


def network_updated(network, eq_state):
    """
    Return a copy of a network whose attributes are updated with an equilibrium state.
    """
    network = network.copy()
    network_update(network, eq_state)

    return network


def network_update(network, eq_state):
    """
    Update in-place the attributes of a network with an equilibrium state.

    TODO: to be extra sure, the node-index and edge-index mappings should be handled
    by EquilibriumModel/EquilibriumStructure
    """
    xyz = eq_state.xyz.tolist()
    lengths = eq_state.lengths.tolist()
    residuals = eq_state.residuals.tolist()
    forces = eq_state.forces.tolist()
    forcedensities = eq_state.force_densities.tolist()
    loads = eq_state.loads.tolist()

    # update q values and lengths on edges
    for idx, edge in network.index_uv().items():
        network.edge_attribute(edge, name="length", value=lengths[idx].pop())
        network.edge_attribute(edge, name="force", value=forces[idx].pop())
        network.edge_attribute(edge, name="q", value=forcedensities[idx])

    # update residuals on nodes
    for idx, node in network.index_key().items():
        for name, value in zip("xyz", xyz[idx]):
            network.node_attribute(node, name=name, value=value)

        for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
            network.node_attribute(node, name=name, value=value)

        for name, value in zip(["px", "py", "pz"], loads[idx]):
            network.node_attribute(node, name=name, value=value)
