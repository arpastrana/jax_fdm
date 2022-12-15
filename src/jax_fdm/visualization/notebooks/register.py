from compas.artists import Artist

from jax_fdm.datastructures import FDNetwork

from jax_fdm.visualization.notebooks import FDNetworkNotebookArtist


__all__ = ["register_artists"]


def register_artists():
    """
    Register objects to the artist factory.
    """
    Artist.register(FDNetwork, FDNetworkNotebookArtist, context="Notebook")
