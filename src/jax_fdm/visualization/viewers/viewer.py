import numpy as _np

# compas_view2 0.7.0 (the legacy COMPAS<2 viewer) uses ``np.int`` in
# ``compas_view2/app/selector.py``, an alias NumPy removed in 1.24. We
# patch it here so the viewer imports under NumPy >= 1.24.
if not hasattr(_np, "int"):
    _np.int = int  # type: ignore[attr-defined]

from compas_view2.app import App

from compas.artists import Artist
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

    def save(self, filepath):
        """
        Save the viewer scene as an image to a filepath.

        Notes
        -----
        The filepath must include the desired image extension.
        The viewer must be called manually after calling this function.
        """
        ext = filepath.split(".")[-1]

        if not self.started:
            self.window.show()

        qimage = self.view.grabFramebuffer()
        qimage.save(filepath, ext)
