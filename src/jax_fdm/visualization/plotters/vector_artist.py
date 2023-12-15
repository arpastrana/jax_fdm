from matplotlib.lines import Line2D
from compas_plotters.artists import LineArtist


class FDVectorPlotterArtist(LineArtist):
    """
    An alternative way to plot vectors as arrows.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_as_segment = True
        self.zorder = 5000
        self.default_linewidth = 0.01

    def draw(self):
        """
        Draw the vector.

        Returns
        -------
        None

        """
        x0, y0 = self.line.start[:2]
        x1, y1 = self.line.end[:2]

        line2d = Line2D(
            [x0, x1],
            [y0, y1],
            linewidth=self.linewidth,
            linestyle=self.linestyle,
            color=self.color,
            zorder=self.zorder,
        )

        self._mpl_line = self.plotter.axes.add_line(line2d)
