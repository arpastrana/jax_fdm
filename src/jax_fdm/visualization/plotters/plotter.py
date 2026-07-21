from typing import Any

from compas_plotter import Plotter as CompasPlotter

__all__ = ["Plotter"]


class Plotter(CompasPlotter):
    """
    A thin wrapper on the `compas_plotter.plotter.Plotter`.

    This object exists for API consistency.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
