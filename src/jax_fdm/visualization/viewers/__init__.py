def is_compasview_installed():
    try:
        import compas_view2  # noqa F401
    except ImportError:
        print("Compas View 2 is not installed and thus jax_fdm.visualization won't be available. Skipping import")
        return False
    return True


if is_compasview_installed():
    from .network import *  # noqa F403
    from .viewer import *  # noqa F403
    from .register import register_artists

    register_artists()

__all__ = [name for name in dir() if not name.startswith('_')]
