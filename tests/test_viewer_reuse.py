"""
The sequential-reuse contract of the viewer wrapper.

One viewer instance serves several show cycles: ``clear()`` between shows
empties the scene, and ``show()`` resets the parent's ``running`` flag on
return so between-show adds stay lightweight. The GL rebuild branch of
``show()`` needs a real display and is exercised by the sequential example
scripts instead.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

compas_viewer = pytest.importorskip("compas_viewer")

from compas.geometry import Sphere  # noqa: E402
from jax_fdm.datastructures import FDNetwork  # noqa: E402
from jax_fdm.visualization import Viewer  # noqa: E402


@pytest.fixture(scope="module")
def viewer():
    # The viewer is a process-wide singleton: one instance for the module.
    return Viewer(width=400, height=300)


@pytest.fixture
def network():
    network = FDNetwork()
    a = network.add_node(x=0.0, y=0.0, z=0.0)
    b = network.add_node(x=1.0, y=0.0, z=0.0)
    network.add_edge(a, b)
    network.node_support(a)
    network.node_support(b)
    network.edge_forcedensity((a, b), -1.0)
    return network


def populate(viewer, network):
    obj = viewer.add(network)
    viewer.add(Sphere(radius=1.0))
    return obj


def test_clear_empties_scene_and_picking_colors(viewer, network):
    obj = populate(viewer, network)
    assert obj in viewer.scene.objects

    viewer.clear()

    assert not list(viewer.scene.objects)
    assert not viewer.scene.instance_colors


def test_clear_then_add_repopulates(viewer, network):
    viewer.clear()
    obj = populate(viewer, network)

    assert obj in viewer.scene.objects
    # the FD network draws through per-category children
    assert {child.name for child in obj.children} == {"Edges", "Reactions", "Loads"}

    viewer.clear()


def test_show_resets_running(viewer, monkeypatch):
    def fake_show(self):
        self.running = True

    monkeypatch.setattr(compas_viewer.Viewer, "show", fake_show)

    viewer.show()

    assert viewer.running is False


def test_show_resets_running_even_when_show_raises(viewer, monkeypatch):
    def broken_show(self):
        self.running = True
        raise RuntimeError("boom")

    monkeypatch.setattr(compas_viewer.Viewer, "show", broken_show)

    with pytest.raises(RuntimeError):
        viewer.show()

    assert viewer.running is False
