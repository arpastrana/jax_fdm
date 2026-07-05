from jax_fdm.visualization.artists.datastructure_artist import FDDatastructureArtist

__all__ = ["FDNetworkArtist"]


class FDNetworkArtist(FDDatastructureArtist):
    """
    The base artist to display a force density network across different contexts.

    It implements the point hooks of :class:`FDDatastructureArtist` in terms of
    the network's ``node_*`` vocabulary and exposes ``network``/``nodes`` aliases
    so existing backends keep working.
    """

    # ==========================================================================
    # Point hooks
    # ==========================================================================

    def _points(self):
        return self.datastructure.nodes()

    def _point_coordinates(self, key):
        return self.datastructure.node_coordinates(key)

    def _point_load(self, key):
        return self.datastructure.node_load(key)

    def _point_reaction(self, key):
        return self.datastructure.node_reaction(key)

    def _point_edges(self, key):
        return self.datastructure.node_edges(key)

    def _point_is_support(self, key):
        return self.datastructure.node_attribute(key, "is_support")

    def _point_label(self, key):
        return f"Node ({key})"

    # ==========================================================================
    # Aliases (network vocabulary)
    # ==========================================================================

    @property
    def network(self):
        return self.datastructure

    @network.setter
    def network(self, network):
        self.datastructure = network

    @property
    def nodes(self):
        return self.points

    @nodes.setter
    def nodes(self, nodes):
        self.points = nodes

    @property
    def node_color(self):
        return self.point_color

    @node_color.setter
    def node_color(self, color):
        self.point_color = color

    @property
    def node_size(self):
        return self.point_size

    @node_size.setter
    def node_size(self, size):
        self.point_size = size

    @property
    def node_xyz(self):
        return self.point_xyz

    @node_xyz.setter
    def node_xyz(self, node_xyz):
        self.point_xyz = node_xyz
