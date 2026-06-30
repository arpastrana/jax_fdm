"""
Characterization tests against an independent structural-analysis solver.

The reference structures are self-stressed and thus have no external loads.
"""

import csv
import os

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm

HERE = os.path.dirname(__file__)
REFERENCE = os.path.join(HERE, "data", "reference")

CASES = ["olympia", "stuttgart21"]


def _reference_network(name):
    """
    Build the reference equilibrium network from its node and edge CSV files.

    Node coordinates are offset by their displacements, edge force densities are
    the axial force over the displaced length, and flagged nodes become supports.
    """
    network = FDNetwork()

    with open(os.path.join(REFERENCE, f"{name}-nodes.csv")) as f:
        for row_num, row in enumerate(csv.reader(f)):
            if row_num == 0:
                continue
            node = int(row[0]) - 1
            network.add_node(node,
                             x=float(row[1]), y=float(row[2]), z=float(row[3]),
                             is_support=(row[4] == "0"),
                             u=float(row[5]), v=float(row[6]), w=float(row[7]))

    with open(os.path.join(REFERENCE, f"{name}-edges.csv")) as f:
        for row_num, row in enumerate(csv.reader(f)):
            if row_num == 0:
                continue
            network.add_edge(int(row[0]) - 1, int(row[1]) - 1, f=float(row[4]))

    for node in network.nodes():
        u, v, w = network.node_attributes(node, ["u", "v", "w"])
        x, y, z = network.node_coordinates(node)
        network.node_attributes(node, "xyz", [x + u, y + v, z + w])
        if network.node_attribute(node, "is_support"):
            network.node_support(node)

    for edge in network.edges():
        force = network.edge_attribute(edge, "f")
        network.edge_forcedensity(edge, force / network.edge_length(*edge))

    return network


@pytest.mark.parametrize("name", CASES)
def test_reference_in_equilibrium(name):
    """
    The reference network satisfies the force density equilibrium equations.
    """
    network = _reference_network(name)

    residual = np.zeros((network.number_of_nodes(), 3))
    for u, v in network.edges():
        q = network.edge_forcedensity((u, v))
        vector = np.array(network.node_coordinates(v)) - np.array(network.node_coordinates(u))
        residual[u] += q * vector
        residual[v] -= q * vector

    free = [node for node in network.nodes() if not network.is_node_support(node)]

    assert jnp.allclose(jnp.asarray(residual[free]), 0.0, atol=1e-7)


@pytest.mark.parametrize("name", CASES)
def test_fdm_reproduces_reference(name):
    """
    Solving the reference network with jax_fdm recovers the reference geometry.
    """
    network = _reference_network(name)

    reference = np.array([network.node_coordinates(node) for node in network.nodes()])
    eq_network = fdm(network, sparse=False)
    solved = np.array([eq_network.node_coordinates(node) for node in eq_network.nodes()])

    assert jnp.allclose(jnp.asarray(solved), jnp.asarray(reference), atol=1e-8)
