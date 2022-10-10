from .artists import *  # noqa F403
from .plotters import *  # noqa F403


def is_compasview_installed():
    try:
        import compas_view2  # noqa F401
        return True
    except ImportError:
        print("Compas View 2 is not installed and thus jax_fdm.visualization won't be available. Skipping import")
        return False


if is_compasview_installed():
    from .viewers import *  # noqa F403
