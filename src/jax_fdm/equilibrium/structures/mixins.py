import jax.numpy as jnp


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

    def _edges_indexed(self):
        """
        An iterable with the edges pointing to the indices of the node keys.
        """
        node_index = self.node_index

        edges_indexed = []
        for u, v in self.edges:
            edge = node_index[int(u)], node_index[int(v)]
            edges_indexed.append(edge)

        return jnp.array(edges_indexed)


class MeshIndexingMixins:
    @property
    def vertex_index(self):
        """
        A dictionary between vertex keys and their enumeration indices.
        """
        return {int(node): index for index, node in enumerate(self.vertices)}

    def _faces_indexed(self):
        """
        An array of the faces pointing to the indices of the node keys.
        """
        vertex_index = self.vertex_index

        findexed = []
        for face in self.faces:
            face_indices = []

            for vertex in face:
                u = int(vertex)
                if u >= 0:
                    index = vertex_index[u]
                else:
                    index = u
                face_indices.append(index)

            findexed.append(tuple(face_indices))

        return jnp.array(findexed)
