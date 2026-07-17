import warnings
from typing import Any

from jax_fdm import has_backend

__all__ = ["has_backend", "null_viewer"]


class _NullObject:
    """
    A null object that absorbs any interaction and returns itself.

    Attribute access, calls, iteration and indexing all succeed and yield a
    null, so an arbitrarily deep expression such as
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

    The returned class warns on construction that the viewer does nothing,
    then behaves as a `_NullObject`: every method call and attribute
    access is a silent no-op. A whole script that builds and drives a viewer
    therefore runs to completion, only warning that nothing was drawn.

    Parameters
    ----------
    name :
        The import name of the missing backend package.

    Returns
    -------
    null_viewer :
        A null viewer class that warns on construction and no-ops thereafter.
    """

    def _init(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            f"The '{name}' backend is not installed. Install it to visualize.",
            stacklevel=2,
        )

    return type("NullViewer", (_NullObject,), {"__init__": _init})
