from compas_plotters.artists import VectorArtist


class FDVectorPlotterArtist(VectorArtist):
    """
    An alternative way to plot vectors as arrows.
    """
    def __init__(
            self,
            vector,
            body_width=0.024,
            head_portion=0.2,
            head_width=0.08,
            width_min=0.0,
            *args,
            **kwargs):
        super().__init__(vector, *args, **kwargs)
        self.width_min = width_min
        self.head_portion = head_portion
        self.head_width = head_width
        self.body_width = body_width

    def draw(self):
        """
        Draw the vector.

        Returns
        -------
        None

        """
        if self.draw_point:
            self._point_artist = self.plotter.add(self.point, edgecolor=self.color)

        min_width = self.width_min
        length = self.vector.length
        width = self.body_width * length + min_width
        head_length = self.head_portion * length + min_width
        head_width = self.head_width * length + min_width

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
            head_length=head_length,
            head_width=head_width,
            color=self.color,
            zorder=self.zorder,
            alpha=1.0,
            lw=0.0,
        )
