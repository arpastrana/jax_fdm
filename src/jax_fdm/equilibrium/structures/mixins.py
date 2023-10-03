# ==========================================================================
# Mixins
# ==========================================================================

class IndexingMixins:
    @property
    def node_index(self):
        """
        A dictionary between node keys and their enumeration indices.
        """
        return {int(node): index for index, node in enumerate(self.nodes)}

    @property
    def edge_index(self):
        """
        A dictionary between edge keys and their enumeration indices.
        """
        return {(int(u), int(v)): index for index, (u, v) in enumerate(self.edges)}

    @property
    def edges_indexed(self):
        """
        An iterable with the edges pointing to the indices of the node keys.
        """
        node_index = self.node_index

        for u, v in self.edges:
            yield node_index[int(u)], node_index[int(v)]


class MeshIndexingMixins:
    @property
    def vertex_index(self):
        """
        A dictionary between vertex keys and their enumeration indices.
        """
        return {int(node): index for index, node in enumerate(self.vertices)}

    @property
    def faces_indexed(self):
        """
        An iterable with the faces pointing to the indices of the node keys.
        """
        vertex_index = self.vertex_index

        for face in self.faces:
            yield tuple(vertex_index[int(u)] for u in face)
