import warnings
from importlib.util import find_spec
from typing import Any

__all__ = ["has_backend", "null_viewer"]


def has_backend(name: str) -> bool:
    """
    Check whether an optional visualization backend is installed.

    The 3D viewer (``compas_viewer``), the notebook viewer (``compas_notebook``)
    and the 2D plotter (``compas_plotter``) are optional dependencies. Their
    absence should degrade gracefully instead of breaking ``import jax_fdm``.

    Parameters
    ----------
    name : str
        The import name of the backend package.

    Returns
    -------
    bool
        ``True`` if the package can be imported, ``False`` otherwise.
    """
    return find_spec(name) is not None


class _NullObject:
    """
    A null object that absorbs any interaction and returns itself.

    Attribute access, calls, iteration, indexing and context management all
    succeed and yield a null, so an arbitrarily deep expression such as
    ``viewer.view.camera.zoom(-35)`` or ``for artist in viewer.artists`` runs
    without error.
    """
    def __getattr__(self, _: str) -> "_NullObject":
        return self

    def __setattr__(self, _: str, __: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_NullObject":
        return self

    def __iter__(self) -> Any:
        return iter(())

    def __getitem__(self, _: Any) -> "_NullObject":
        return self

    def __setitem__(self, _: Any, __: Any) -> None:
        pass


def null_viewer(name: str) -> type:
    """
    Build an inert viewer class to use when its backend is not installed.

    The returned class warns once on construction that the viewer does nothing,
    then behaves as a :class:`_NullObject`: every method call and attribute
    access is a silent no-op. A whole script that builds and drives a viewer
    therefore runs to completion, only warning that nothing was drawn.

    Parameters
    ----------
    name : str
        The import name of the missing backend package.

    Returns
    -------
    type
        A null viewer class that warns on construction and no-ops thereafter.
    """
    def _init(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(f"The '{name}' backend is not installed. "
                      "Install it to visualize.", stacklevel=2)

    return type("NullViewer", (_NullObject,), {"__init__": _init})
