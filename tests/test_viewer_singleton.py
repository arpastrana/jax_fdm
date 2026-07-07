import pytest

compas_viewer = pytest.importorskip("compas_viewer")

from compas_viewer.singleton import Singleton  # noqa: E402
from compas_viewer.singleton import SingletonMeta  # noqa: E402

from jax_fdm.visualization.viewers.viewer import ViewerMeta  # noqa: E402
from jax_fdm.visualization.viewers.viewer import retire_viewer  # noqa: E402


class StubTimer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


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


def test_retire_viewer_is_defensive():
    retire_viewer(object())
