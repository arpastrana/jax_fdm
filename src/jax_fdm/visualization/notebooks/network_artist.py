from jax_fdm.visualization.artists import FDNetworkArtist
from jax_fdm.visualization.notebooks.datastructure_artist import FDDatastructureNotebookArtist

__all__ = ["FDNetworkNotebookArtist"]


class FDNetworkNotebookArtist(FDDatastructureNotebookArtist, FDNetworkArtist):
    """
    An artist that draws a force density network to a :class:`compas_notebook.viewer.Viewer`.

    It pairs the batched notebook machinery of :class:`FDDatastructureNotebookArtist`
    with the network's ``node_*`` vocabulary via :class:`FDNetworkArtist`.
    """
    pass
