from compas.artists import Artist

from jax_fdm.datastructures import FDNetwork

from jax_fdm.visualization.plotters import FDNetworkPlotterArtist
from jax_fdm.visualization.plotters import FDVector
from jax_fdm.visualization.plotters import FDVectorPlotterArtist


__all__ = ["register_artists"]


def register_artists():
    """
    Register objects to the artist factory.
    """
    Artist.register(FDVector, FDVectorPlotterArtist, context="Plotter")
    Artist.register(FDNetwork, FDNetworkPlotterArtist, context="Plotter")
