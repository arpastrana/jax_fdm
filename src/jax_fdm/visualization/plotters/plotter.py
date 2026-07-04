from compas_plotters import Plotter as CompasPlotter


class Plotter(CompasPlotter):
    """
    A thin wrapper on the :class:`compas_plotters.plotter.Plotter`.

    This object exists for API consistency.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
