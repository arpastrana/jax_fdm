from math import radians

from compas.artists import Artist

from compas_notebook.app import App as NotebookApp

from jax_fdm.datastructures import FDNetwork


__all__ = ["NotebookViewer"]


class NotebookViewer(NotebookApp):
    """
    A thin wrapper on :class:`compas_notebook.app.App`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.artists = []

    def add(self, data, **kwargs):
        """
        Add a COMPAS data object to the notebook viewer.

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
            return super(NotebookViewer, self).add(data, **kwargs)

        if kwargs.get("as_wireframe"):
            del kwargs["as_wireframe"]
            return super(NotebookViewer, self).add(data, **kwargs)

        artist = Artist(data, viewer=self, context="Notebook", **kwargs)
        self.artists.append(artist)
        artist.draw()
        artist.add()

    def show(self, *args, **kwargs):
        """
        Display the current scene.
        """
        self.set_camera(angles=[radians(89.), 0., 0.])
        return super(NotebookViewer, self).show(viewer="notebook", *args, **kwargs)
