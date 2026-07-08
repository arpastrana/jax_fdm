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
from jax_fdm.visualization.viewers.scene_objects import FDGroupObject  # noqa: E402
from jax_fdm.visualization.viewers.scene_objects import FDMeshObject  # noqa: E402
from jax_fdm.visualization.viewers.scene_objects import FDNetworkObject  # noqa: E402
from jax_fdm.visualization.viewers.sidebar import FDObjectSetting  # noqa: E402

# Constructing a scene object lazily creates the process-wide compas_viewer
# singleton. Create the jax_fdm wrapper first, so tests running later in the
# same process (e.g. test_viewer_reuse.py) get the wrapper class and not a
# bare compas_viewer.Viewer pinned under the shared singleton key.
VIEWER = Viewer(width=400, height=300)


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
    return {child.name: child.build_soup()
            for child in obj.children if hasattr(child, "build_soup")}


def category_groups(obj):
    return {child.name: child for child in obj.children
            if isinstance(child, FDGroupObject)}


def category_candidates(obj):
    """
    The candidate keys of every category, in fused soup order.
    """
    return {"Edges": obj.edges,
            obj.points_name: obj.points,
            "Reactions": obj.reaction_points,
            "Loads": obj.load_points}


def fused_slots(fused_obj):
    """
    The fused category soups, split into one equal block per candidate key.

    Every element of a category renders the same template, so the fused soup
    is the per-key blocks concatenated in candidate order.
    """
    candidates = category_candidates(fused_obj)

    slots = {}
    for name, (positions, colors) in category_soups(fused_obj).items():
        keys = candidates[name]
        slots[name] = {key: (block, colorblock) for key, block, colorblock
                       in zip(keys, np.split(positions, len(keys)), np.split(colors, len(keys)))}
    return slots


def assert_elements_match_fused_slots(unfused_obj, fused_obj):
    """
    Every element child's soup must equal its slot of the fused category soup.
    """
    slots = fused_slots(fused_obj)
    groups = category_groups(unfused_obj)
    assert groups.keys() == slots.keys()

    for name, group in groups.items():
        for child in group.children:
            positions, colors = child.build_soup()
            expected_positions, expected_colors = slots[name][child.key]
            np.testing.assert_array_equal(positions, expected_positions, err_msg=child.name)
            np.testing.assert_array_equal(colors, expected_colors, err_msg=child.name)


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
    assert stale, "no category soups collected: expected a fused scene object"

    for name in rebuilt:
        for got, expected in zip(stale[name], rebuilt[name]):
            np.testing.assert_array_equal(got, expected, err_msg=name)


def move_network(network):
    for node in network.nodes():
        x, y, z = network.node_coordinates(node)
        network.node_attributes(node, names="xyz", values=(x, y, z + 0.5 * node))
    network.edge_attribute((1, 2), "force", -2.0)


def move_mesh(mesh):
    for vertex in mesh.vertices():
        x, y, z = mesh.vertex_coordinates(vertex)
        mesh.vertex_attributes(vertex, names="xyz", values=(x, y, z + 0.3 * vertex))
    mesh.edge_attribute((0, 1), "force", 2.0)


def test_network_update_matches_rebuild(network):
    assert_update_matches_rebuild(
        lambda: FDNetworkObject(item=network, context="Viewer", fuse=True, **NETWORK_KWARGS),
        network, move_network)


def test_mesh_update_matches_rebuild(mesh):
    assert_update_matches_rebuild(
        lambda: FDMeshObject(item=mesh, context="Viewer", fuse=True, **MESH_KWARGS),
        mesh, move_mesh)


def test_unfused_elements_match_fused_soups(network, mesh):
    """
    Fused and per-element render paths must be vertex-identical: every
    element child's soup equals its slot of the fused category soup.
    """
    for cls, datastructure, kwargs in (
        (FDNetworkObject, network, NETWORK_KWARGS),
        (FDMeshObject, mesh, MESH_KWARGS),
    ):
        unfused = cls(item=datastructure, context="Viewer", **kwargs)
        fused = cls(item=datastructure, context="Viewer", fuse=True, **kwargs)
        assert_elements_match_fused_slots(unfused, fused)


def test_unfused_update_matches_rebuild(network, mesh):
    """
    After mutating geometry and updating, every element child's soup must
    equal its slot of a fused scene object built from scratch (element
    membership is frozen at add, so pruning is not re-evaluated).
    """
    for cls, datastructure, kwargs, move in (
        (FDNetworkObject, network, NETWORK_KWARGS, move_network),
        (FDMeshObject, mesh, MESH_KWARGS, move_mesh),
    ):
        unfused = cls(item=datastructure, context="Viewer", **kwargs)

        move(datastructure)
        unfused.update()

        fresh_fused = cls(item=datastructure, context="Viewer", fuse=True, **kwargs)
        assert_elements_match_fused_slots(unfused, fresh_fused)


def test_unfused_tree_shape(network):
    """
    The default scene object is a three-level tree: parent, one group per
    category, one named child per element. Arrows below tolerance at add
    time are pruned.
    """
    obj = FDNetworkObject(item=network, context="Viewer", **NETWORK_KWARGS)

    groups = category_groups(obj)
    assert set(groups) == {"Edges", "Nodes", "Loads", "Reactions"}

    assert [child.key for child in groups["Edges"].children] == obj.edges
    assert [child.key for child in groups["Nodes"].children] == obj.points
    assert [child.name for child in groups["Edges"].children] == [f"Edge {edge}" for edge in obj.edges]
    assert [child.name for child in groups["Nodes"].children] == [f"Node {node}" for node in obj.points]

    # Only node 1 is loaded; only node 0 carries a reaction above tolerance.
    assert [child.name for child in groups["Loads"].children] == ["Load 1"]
    assert [child.name for child in groups["Reactions"].children] == ["Reaction 0"]

    for group in groups.values():
        for child in group.children:
            assert child.fd_parent is obj


def test_sidebar_readout_installed_and_populates(network):
    """
    The viewer installs the force density object settings tab, and selecting
    an element child populates it with a read-only attribute tree.
    """
    assert isinstance(VIEWER.ui.sidebar.object_setting, FDObjectSetting)

    obj = VIEWER.add(network, **NETWORK_KWARGS)
    setting = VIEWER.ui.sidebar.object_setting

    edge_child = category_groups(obj)["Edges"].children[0]
    edge_child.is_selected = True
    try:
        setting.update()
        form = setting.children[0]
        nodes = [node for node in form.tree.traverse() if not node.is_root]
        assert nodes[0].name == "Edge (0, 1)"
        attributes = {node.name: node.attributes.get("value") for node in nodes[1:]}
        assert set(attributes) == {"q", "force", "length"}
        assert attributes["q"] == "-1"
    finally:
        edge_child.is_selected = False
        VIEWER.clear()


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
        assert set(obj.adjacency) == set(obj.points)
        for point, edges in obj.adjacency.items():
            assert sorted(edges) == sorted(obj.point_edges(point))
