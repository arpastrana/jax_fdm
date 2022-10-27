from compas.artists import Artist

try:
    from compas_view2.app import App
except ImportError:
    class App:
        pass

from jax_fdm.datastructures import FDNetwork


__all__ = ["Viewer"]


class Viewer(App):
    """
    A thin wrapper on :class:`compas_view2.app.App`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.artists = []

    def add(self, data, **kwargs):
        """
        Add a COMPAS data object to the viewer.

        It adds a viewer argument if the object is a :class:`jax_fdm.datastructures.FDNetwork`

        Parameters
        ----------
        data: :class:`compas.geometry.Primitive` | :class:`compas.geometry.Shape` | :class:`compas.datastructures.Datastructure`
            A COMPAS data object.
        **kwargs : dict, optional
            Additional visualization options.

        Returns
        -------
        view_data :class:`compas_view2.objects.Object`
            A visualization object.
        """
        if not isinstance(data, (FDNetwork)):
            return super(Viewer, self).add(data, **kwargs)

        if kwargs.get("as_wireframe"):
            del kwargs["as_wireframe"]
            return super(Viewer, self).add(data, **kwargs)

        artist = Artist(data, viewer=self, context="Viewer", **kwargs)
        self.artists.append(artist)
        artist.draw()
        artist.add()
