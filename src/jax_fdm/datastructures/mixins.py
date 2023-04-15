"""
Helping meshes to become networks and the other way around.
"""

class NodeMixins:
    """
    Wrapper methods that convert node-based queries to vertex-based ones.
    """
    def nodes(self, data=False):
        """
        Yield the vertices of the mesh.
        """
        return self.vertices(data)

    def nodes_where(self, condition):
        """
        yield the vertices where condition holds true.
        """
        return self.vertices_where(condition)

    def node_coordinates(self, node, axes):
        """
        The vertex XYZ coordinates.
        """
        return self.vertex_coordinates(node, axes)

    def node_attribute(self, node, name, value=None):
        """
        Gets or sets a vertex attribute.
        """
        return self.vertex_attribute(node, name, value)

    def node_attributes(self, node, names=None, values=None):
        """
        Gets or sets a vertex attribute.
        """
        return self.vertex_attributes(node, names, values)

    def nodes_attributes(self, names=None, values=None, keys=None):
        """
        Gets or sets a vertex attribute.
        """
        return self.vertices_attributes(names, values, keys)
