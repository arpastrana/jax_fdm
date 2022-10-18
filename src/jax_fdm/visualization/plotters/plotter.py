from compas_plotters import Plotter


class Plotter(Plotter):
    """
    A thin wrapper on the :class:`compas_plotters.plotter.Plotter`.

    This object exists for API consistency.
    """
    def __init__(self, *args, **kwargs):
        super(Plotter, self).__init__(*args, **kwargs)
