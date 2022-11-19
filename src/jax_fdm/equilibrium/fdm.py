import numpy as np

from jax_fdm import DTYPE_NP

from jax_fdm.equilibrium import EquilibriumModel


# ==========================================================================
# Form-finding
# ==========================================================================

def _fdm(network, q, xyz_fixed, loads):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    model = EquilibriumModel(network)

    # compute static equilibrium
    eq_state = model(q, xyz_fixed, loads)

    # update equilibrium state in a copy of the network
    return network_updated(network, eq_state)


def fdm(network):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    params = (np.array(p, dtype=DTYPE_NP) for p in network.parameters())

    return _fdm(network, *params)


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
    model = EquilibriumModel(network)

    opt_problem = optimizer.problem(model, loss, parameters, constraints, maxiter, tol, callback)
    opt_params = optimizer.solve(opt_problem)
    q, xyz_fixed, loads = optimizer.parameters_fdm(opt_params)

    return _fdm(network, q, xyz_fixed, loads)


# ==========================================================================
# Helpers
# ==========================================================================

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
