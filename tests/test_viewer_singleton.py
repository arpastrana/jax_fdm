import pytest

compas_viewer = pytest.importorskip("compas_viewer")

from compas_viewer.singleton import Singleton  # noqa: E402
from compas_viewer.singleton import SingletonMeta  # noqa: E402

from jax_fdm.visualization.viewers.viewer import ViewerMeta  # noqa: E402
from jax_fdm.visualization.viewers.viewer import retire_viewer  # noqa: E402
from jax_fdm.visualization.viewers.viewer import stop_watch_timers  # noqa: E402


class StubTimer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class StubBoundComponent:
    """A stand-in for a sidebar component with a change-watch timer."""
    def __init__(self):
        self.watching = True

    def stop_watching(self):
        self.watching = False


class StubContainer:
    def __init__(self, *children):
        self.children = list(children)


class StubTabform:
    def __init__(self, **tabs):
        self.tabs = tabs


class StubUI:
    def __init__(self, sidebar):
        self.sidebar = sidebar


def make_sidebar():
    """A sidebar tree shaped like compas_viewer's: container > tabform > container > bound components."""
    sceneform_edit = StubBoundComponent()
    object_edit = StubBoundComponent()
    camera_edit = StubBoundComponent()
    tabform = StubTabform(Object=StubContainer(object_edit),
                          Camera=StubContainer(camera_edit))
    sidebar = StubContainer(StubContainer(sceneform_edit), tabform)
    return sidebar, [sceneform_edit, object_edit, camera_edit]


class FakeViewer(Singleton, metaclass=ViewerMeta):
    """A stand-in with the singleton lifecycle attributes, but no Qt."""
    def __init__(self):
        self.running = False
        self._spent = False
        self.timer = StubTimer()
        self.inits = getattr(self, "inits", 0) + 1


@pytest.fixture(autouse=True)
def clean_singleton_cache():
    SingletonMeta._instances.pop(FakeViewer, None)
    yield
    SingletonMeta._instances.pop(FakeViewer, None)


def test_keeps_live_unspent_instance():
    viewer = FakeViewer()
    again = FakeViewer()

    assert again is viewer
    assert again.inits == 1


def test_evicts_running_instance():
    viewer = FakeViewer()
    viewer.running = True
    timer = viewer.timer

    fresh = FakeViewer()

    assert fresh is not viewer
    assert timer.stopped
    assert viewer.running is False
    assert fresh.running is False


def test_evicts_spent_instance():
    viewer = FakeViewer()
    viewer._spent = True

    fresh = FakeViewer()

    assert fresh is not viewer


def test_evicts_foreign_instance():
    class Foreign(Singleton, metaclass=SingletonMeta):
        pass

    foreign = Foreign()
    SingletonMeta._instances[FakeViewer] = foreign

    fresh = FakeViewer()

    assert isinstance(fresh, FakeViewer)
    assert fresh is not foreign


def test_eviction_stops_sidebar_watch_timers():
    viewer = FakeViewer()
    viewer._spent = True
    sidebar, components = make_sidebar()
    viewer.ui = StubUI(sidebar)

    FakeViewer()

    assert all(not component.watching for component in components)


def test_stop_watch_timers_walks_containers_and_tabs():
    sidebar, components = make_sidebar()

    stop_watch_timers(sidebar)

    assert all(not component.watching for component in components)


def test_retire_viewer_is_defensive():
    retire_viewer(object())
