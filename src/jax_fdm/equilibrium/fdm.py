from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


# ==========================================================================
# Form-finding
# ==========================================================================

def _fdm(model, structure):
    """
    Generate a network in a state of static equilibrium using the force density method.
    """
    # compute static equilibrium
    eq_state = model(structure)

    # update equilibrium state in a copy of the network
    return network_updated(structure.network, eq_state)


def fdm(network):
    """
    Generate a network in a state of static equilibrium using the force density method.
    """
    network_validate(network)

    model = EquilibriumModel.from_network(network)
    structure = EquilibriumStructure.from_network(network)

    return _fdm(model, structure)


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
                    callback=None):
    """
    Generate a network in a constrained state of static equilibrium using the force density method.
    """
    network_validate(network)

    model = EquilibriumModel.from_network(network)
    structure = EquilibriumStructure.from_network(network)

    opt_problem = optimizer.problem(model, structure, loss, parameters, constraints, maxiter, tol, callback)
    opt_params = optimizer.solve(opt_problem)
    # q, xyz_fixed, loads = optimizer.parameters_fdm(opt_params)
    opt_model = optimizer.parameters_fdm(opt_params)

    return _fdm(opt_model, structure)


# ==========================================================================
# Helpers
# ==========================================================================

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
