"""
The force density scene objects of the 2D plotter backend.

Dispatch through the compas scene registry, force-density edge styling on the
matplotlib collections, load and reaction arrows as batched patch collections,
and the clear-and-redraw cycle.
"""
import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

compas_plotters = pytest.importorskip("compas_plotters")

from compas_plotters.scene import GraphObject  # noqa: E402
from compas_plotters.scene import MeshObject  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402

from compas.datastructures import Graph  # noqa: E402
from compas.datastructures import Mesh  # noqa: E402
from jax_fdm.datastructures import FDMesh  # noqa: E402
from jax_fdm.datastructures import FDNetwork  # noqa: E402
from jax_fdm.equilibrium import fdm  # noqa: E402
from jax_fdm.visualization import Plotter  # noqa: E402
from jax_fdm.visualization.plotters import FDMeshPlotterObject  # noqa: E402
from jax_fdm.visualization.plotters import FDNetworkPlotterObject  # noqa: E402
from jax_fdm.visualization.style import COLOR_COMPRESSION  # noqa: E402
from jax_fdm.visualization.style import COLOR_SUPPORT  # noqa: E402

# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def network():
    """A solved arch network in the XY plane, the plane the plotter projects to."""
    network = FDNetwork()

    num_nodes = 6
    for i in range(num_nodes):
        network.add_node(x=float(i), y=0.0, z=0.0)
    for i in range(num_nodes - 1):
        network.add_edge(i, i + 1)

    network.node_support(0)
    network.node_support(num_nodes - 1)
    network.edges_forcedensities(-1.0, keys=list(network.edges()))
    network.nodes_loads([0.0, -0.2, 0.0], keys=list(range(1, num_nodes - 1)))

    return fdm(network)


@pytest.fixture
def mesh():
    """A solved meshgrid strip, supported on the boundary."""
    mesh = FDMesh.from_meshgrid(dx=3, nx=3)

    for vertex in mesh.vertices_on_boundary():
        mesh.vertex_support(vertex)
    mesh.edges_forcedensities(-2.0, keys=list(mesh.edges()))
    mesh.vertices_loads([0.0, 0.0, -0.5], keys=list(mesh.vertices_free()))

    return fdm(mesh)


@pytest.fixture
def plotter():
    return Plotter()


# ==========================================================================
# Dispatch
# ==========================================================================

def test_network_dispatches_to_fd_object(plotter, network):
    obj = plotter.add(network)
    assert type(obj) is FDNetworkPlotterObject


def test_mesh_dispatches_to_fd_object(plotter, mesh):
    obj = plotter.add(mesh)
    assert type(obj) is FDMeshPlotterObject


def test_plain_compas_types_dispatch_upstream(plotter):
    graph = Graph()
    a = graph.add_node(x=0.0, y=0.0, z=0.0)
    b = graph.add_node(x=1.0, y=1.0, z=0.0)
    graph.add_edge(a, b)

    assert type(plotter.add(graph)) is GraphObject
    assert type(plotter.add(Mesh.from_meshgrid(dx=1, nx=1))) is MeshObject


# ==========================================================================
# Edge styling
# ==========================================================================

def edge_collection(obj):
    return next(o for o in obj._mpl_objects if isinstance(o, LineCollection))


def test_edge_colors_by_force(plotter, network):
    obj = plotter.add(network, edgecolor="force", show_reactions=False, show_loads=False)

    colors = edge_collection(obj).get_colors()
    # the arch is fully compressive
    assert all(tuple(color[:3]) == COLOR_COMPRESSION.rgb for color in colors)


def test_edge_widths_remap_forces(plotter, network):
    obj = plotter.add(network, edgewidth=(1.0, 3.0))

    widths = edge_collection(obj).get_linewidths()
    assert len(widths) == network.number_of_edges()
    assert min(widths) >= 1.0 and max(widths) <= 3.0
    # the outermost edges carry more force than the crown
    assert max(widths) > min(widths)


def test_edge_width_scalar_broadcasts(plotter, network):
    obj = plotter.add(network, edgewidth=2.5)

    widths = set(float(w) for w in edge_collection(obj).get_linewidths())
    assert widths == {2.5}


def test_edge_width_dict_passes_through(plotter, network):
    edge = next(iter(network.edges()))
    widths = {e: 4.0 if e == edge else 1.0 for e in network.edges()}
    obj = plotter.add(network, edgewidth=widths)

    assert sorted(set(float(w) for w in edge_collection(obj).get_linewidths())) == [1.0, 4.0]


# ==========================================================================
# Arrows
# ==========================================================================

def arrow_collections(obj):
    return [o for o in obj._mpl_objects if isinstance(o, PatchCollection)]


def test_arrow_counts(plotter, network):
    obj = plotter.add(network, show_nodes=False)

    # nodes off: loads and reactions are the only patch collections
    loads, reactions = arrow_collections(obj)
    assert len(loads.get_paths()) == 4  # the four loaded free nodes
    assert len(reactions.get_paths()) == 2  # the two supports


def test_arrow_visibility_flags(plotter, network):
    obj = plotter.add(network, show_nodes=False, show_loads=False, show_reactions=False)
    assert not arrow_collections(obj)


def test_below_tolerance_arrows_are_skipped(plotter, network):
    obj = plotter.add(network, show_nodes=False, loadtol=1e6, show_reactions=False)
    assert not arrow_collections(obj)


def test_reaction_color_defaults_green_under_force_coloring(plotter, network):
    from jax_fdm.visualization.style import reaction_color_default

    obj = plotter.add(network, edgecolor="force")
    assert obj.reaction_color == reaction_color_default("force")


# ==========================================================================
# Supports
# ==========================================================================

def test_support_nodes_are_green(plotter, network):
    obj = plotter.add(network, show_nodes=True)

    for node in network.nodes():
        expected = COLOR_SUPPORT if network.is_node_support(node) else obj.nodecolor[node]
        assert obj.nodecolor[node] == expected
    assert obj.nodecolor[0] == COLOR_SUPPORT
    assert obj.nodecolor[1] != COLOR_SUPPORT


# ==========================================================================
# Redraw
# ==========================================================================

def test_redraw_replaces_artists(plotter, network):
    obj = plotter.add(network, show_nodes=True)
    before = list(obj._mpl_objects)

    network.edges_forcedensities(-5.0, keys=list(network.edges()))
    obj.redraw()

    assert len(obj._mpl_objects) == len(before)
    assert not set(obj._mpl_objects) & set(before)
    for artist in before:
        assert artist.axes is None  # removed from the canvas


# ==========================================================================
# Zoom extents
# ==========================================================================

def test_viewdata_covers_arrows(plotter, network):
    obj = plotter.add(network, reactionscale=10.0)

    xs = [x for x, y in obj.viewdata()]
    node_xs = [network.node_coordinates(node)[0] for node in network.nodes()]
    # the scaled-up reaction arrows extend beyond the nodes
    assert min(xs) < min(node_xs) or max(xs) > max(node_xs)


# ==========================================================================
# Mesh faces
# ==========================================================================

def test_mesh_draws_faces_by_default(plotter, mesh):
    obj = plotter.add(mesh, show_loads=False, show_reactions=False)
    assert len(arrow_collections(obj)) == 1  # the face patch collection


def test_mesh_faces_can_be_hidden(plotter, mesh):
    obj = plotter.add(mesh, show_faces=False, show_loads=False, show_reactions=False)
    assert not arrow_collections(obj)
