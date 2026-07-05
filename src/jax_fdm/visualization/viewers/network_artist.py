from jax_fdm.visualization.artists import FDNetworkArtist
from jax_fdm.visualization.viewers.datastructure_artist import FDDatastructureViewerArtist

__all__ = ["FDNetworkViewerArtist"]


class FDNetworkViewerArtist(FDDatastructureViewerArtist, FDNetworkArtist):
    """
    An artist that draws a force density network to a :class:`compas_viewer.Viewer`.

    All the scene machinery lives in :class:`FDDatastructureViewerArtist`; this
    subclass only pairs it with the network's ``node_*`` vocabulary (via
    :class:`FDNetworkArtist`) and labels its points subgroup "Nodes".
    """
    points_group_name = "Nodes"

    # Back-compat aliases for the network vocabulary.
    @property
    def viewer_nodes(self):
        return self.viewer_points

    @viewer_nodes.setter
    def viewer_nodes(self, viewer_nodes):
        self.viewer_points = viewer_nodes
