from .artists import *  # noqa F403
from .plotters import *  # noqa F403

try:
    from .viewers import *  # noqa F403
    from .register import *  # noqa F403
except (ImportError, ModuleNotFoundError):
    print("Compas View 2 is not installed and thus jax_fdm.visualization won't be available. Skipping import")
    pass
