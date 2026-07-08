"""
The update contract of the force density scene objects.

The scene objects freeze topology at add time (soup membership, point-edge
adjacency) and re-derive everything else from the live datastructure on
``update()``. These tests pin cache coherence: after mutating geometry and
updating, every category soup must equal the soup of a scene object built
from scratch on the same datastructure.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("compas_viewer")

from jax_fdm.datastructures import FDMesh  # noqa: E402
from jax_fdm.datastructures import FDNetwork  # noqa: E402
from jax_fdm.visualization import Viewer  # noqa: E402
from jax_fdm.visualization.viewers.scene_objects import FDMeshObject  # noqa: E402
from jax_fdm.visualization.viewers.scene_objects import FDNetworkObject  # noqa: E402

# Constructing a scene object lazily creates the process-wide compas_viewer
# singleton. Create the jax_fdm wrapper first, so tests running later in the
# same process (e.g. test_viewer_reuse.py) get the wrapper class and not a
# bare compas_viewer.Viewer pinned under the shared singleton key.
_ = Viewer(width=400, height=300)


@pytest.fixture
def network():
    network = FDNetwork()
    for key, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 1.0), (3.0, 0.0, 0.0)]):
        network.add_node(key=key, x=x, y=y, z=z)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        network.add_edge(u, v)
        network.edge_forcedensity((u, v), -1.0)
        network.edge_attribute((u, v), "force", 1.0)
    network.node_support(0)
    network.node_support(3)
    network.node_load(1, [0.0, 0.0, -1.0])
    network.node_attributes(0, names=("rx", "ry", "rz"), values=(0.5, 0.0, 0.5))
    return network


@pytest.fixture
def mesh():
    mesh = FDMesh.from_vertices_and_faces(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [[0, 1, 2, 3]],
    )
    mesh.vertex_support(0)
    mesh.vertex_load(2, [0.0, 0.0, -1.0])
    mesh.vertex_attributes(0, names=("rx", "ry", "rz"), values=(0.3, 0.0, 0.4))
    for edge in mesh.edges():
        mesh.edge_forcedensity(edge, -1.0)
        mesh.edge_attribute(edge, "force", -1.0)
    return mesh


STYLE_KWARGS = {
    "edgecolor": "force",
    "edgewidth": (0.05, 0.25),
    "show_loads": True,
    "show_reactions": True,
}

# Each datastructure styles its points in its own vocabulary.
NETWORK_KWARGS = {"show_nodes": True, **STYLE_KWARGS}
MESH_KWARGS = {"show_vertices": True, **STYLE_KWARGS}


def category_soups(obj):
    return {child.name: child._build_soup()
            for child in obj.children if hasattr(child, "_build_soup")}


def assert_update_matches_rebuild(factory, datastructure, move):
    """
    After mutating geometry and updating, every category soup must equal
    the soup of a scene object built from scratch.
    """
    obj = factory()

    move(datastructure)
    obj.update()

    fresh = factory()

    stale, rebuilt = category_soups(obj), category_soups(fresh)
    assert stale.keys() == rebuilt.keys()

    for name in rebuilt:
        for got, expected in zip(stale[name], rebuilt[name]):
            np.testing.assert_array_equal(got, expected, err_msg=name)


def test_network_update_matches_rebuild(network):
    def move(network):
        for node in network.nodes():
            x, y, z = network.node_coordinates(node)
            network.node_attributes(node, names="xyz", values=(x, y, z + 0.5 * node))
        network.edge_attribute((1, 2), "force", -2.0)

    assert_update_matches_rebuild(
        lambda: FDNetworkObject(item=network, context="Viewer", **NETWORK_KWARGS),
        network, move)


def test_mesh_update_matches_rebuild(mesh):
    def move(mesh):
        for vertex in mesh.vertices():
            x, y, z = mesh.vertex_coordinates(vertex)
            mesh.vertex_attributes(vertex, names="xyz", values=(x, y, z + 0.3 * vertex))
        mesh.edge_attribute((0, 1), "force", 2.0)

    assert_update_matches_rebuild(
        lambda: FDMeshObject(item=mesh, context="Viewer", **MESH_KWARGS),
        mesh, move)


def test_point_kwargs_speak_the_datastructure_vocabulary(network, mesh):
    """
    A network styles its points as nodes, a mesh as vertices; the vocabulary
    maps onto the shared point state and spawns the sphere category child.
    """
    obj = FDNetworkObject(item=network, context="Viewer", show_nodes=True, nodesize=0.4)
    assert all(size == 0.4 for size in obj.point_size.values())
    assert "Nodes" in {child.name for child in obj.children}

    obj = FDMeshObject(item=mesh, context="Viewer", show_vertices=True, vertexsize=0.4)
    assert all(size == 0.4 for size in obj.point_size.values())
    assert "Vertices" in {child.name for child in obj.children}


def test_adjacency_is_frozen_and_complete(network, mesh):
    for obj in (
        FDNetworkObject(item=network, context="Viewer"),
        FDMeshObject(item=mesh, context="Viewer"),
    ):
        assert set(obj._adjacency) == set(obj.points)
        for point, edges in obj._adjacency.items():
            assert sorted(edges) == sorted(obj.point_edges(point))
