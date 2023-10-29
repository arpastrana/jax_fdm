from compas_plotters.artists import VectorArtist


class FDVectorPlotterArtist(VectorArtist):
    """
    An alternative way to plot vectors as arrows.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw(self):
        """
        Draw the vector.

        Returns
        -------
        None

        """
        if self.draw_point:
            self._point_artist = self.plotter.add(self.point, edgecolor=self.color)

        length = self.vector.length
        min_width = 0.01
        width = length * 0.012 + min_width
        x, y = self.point[:2]
        dx, dy = (self.vector)[:2]
        self.zorder = 5000

        self._mpl_vector = self.plotter.axes.arrow(
            x,
            y,
            dx,
            dy,
            width=width,
            length_includes_head=True,
            head_width=length * 0.06 + min_width,
            head_length=length * 0.12 + min_width,
            color=self.color,
            zorder=self.zorder,
            alpha=1.0,
            lw=0.0,
        )
