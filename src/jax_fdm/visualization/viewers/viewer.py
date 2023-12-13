import os
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
        kwargs = kwargs or {}
        if "viewmode" not in kwargs:
            kwargs["viewmode"] = "lighted"
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

    # def save(self, filename="scene.png", filedir=None):
    def save(self, filepath):
        """
        Save the scene as an image.

        Returns
        -------
        None
            The objects are updated in place
        """
        ext = filepath.split(".")[-1]
        assert ext == "png"

        if not self.started:
            self.window.show()

        qimage = self.view.grabFramebuffer()
        filepath = os.path.abspath(filepath)
        qimage.save(filepath, ext)

        print(f"Saved viewer scene at {filepath}")
        # self._app.quit()
