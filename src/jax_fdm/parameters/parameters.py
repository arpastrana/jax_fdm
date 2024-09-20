from jax.numpy import inf


# ==========================================================================
# Parameter
# ==========================================================================

class Parameter:
    """
    The base class for all parameters.

    Parameters
    ----------
    key : `int` | `Tuple[int]`
        The key in the network of the parameter.
    bound_low : `float`, optional
        The lower bound of this parameter for optimization.
        Defaults to `-inf`.
    bound_ip : `float`, optional
        The upper bound of this parameter for optimization.
        Defaults to `+inf`.
    """
    def __init__(self, key, bound_low=None, bound_up=None):
        """
        Initialize the parameter.
        """
        self.key = key
        self.attr_name = None

        self._bound_low = None
        self.bound_low = bound_low

        self._bound_up = None
        self.bound_up = bound_up

    @property
    def bound_low(self):
        """
        The lower bound of the parameter.
        """
        return self._bound_low

    @property
    def bound_up(self):
        """
        The upper bound of the parameter.
        """
        return self._bound_up

    @bound_low.setter
    def bound_low(self, value):
        if value is None:
            value = -inf
        self._bound_low = value

    @bound_up.setter
    def bound_up(self, value):
        if value is None:
            value = inf
        self._bound_up = value

    def index(self, model):
        """
        Get the index of the parameter key in the structure of a model.
        """
        raise NotImplementedError

    def value(self, model, network):
        """
        Get the current value of a paramter from the structure of a model.
        """
        raise NotImplementedError


# ==========================================================================
# Individual parameters
# ==========================================================================

class NodeParameter(Parameter):
    """
    A node parameter.
    """
    def index(self, model, structure):
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        return structure.node_index[self.key]

    def value(self, model, network):
        """
        Get the current value of the node parameter.
        """
        return network.node_attribute(self.key, name=self.attr_name)


class VertexParameter(Parameter):
    """
    A vertex parameter.
    """
    def index(self, model, structure):
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        return structure.vertex_index[self.key]

    def value(self, model, mesh):
        """
        Get the current value of the node parameter.
        """
        return mesh.vertex_attribute(self.key, name=self.attr_name)


class EdgeParameter(Parameter):
    """
    An edge parameter.
    """
    def index(self, model, structure):
        """
        Get the index of the key of the edge parameter in the structure of a model.
        """
        return structure.edge_index[self.key]

    def value(self, model, datastructure):
        """
        Get the current value of the edge parameter.
        """
        return datastructure.edge_attribute(self.key, name=self.attr_name)


# ==========================================================================
# Parameter groups
# ==========================================================================

class ParameterGroup(Parameter):
    """
    A parent class for groups of parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.key) > 0


class NodeGroupParameter(ParameterGroup):
    """
    A single parameter applied to a group of nodes.
    """
    def index(self, model, structure):
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [structure.node_index[key] for key in self.key]

    def value(self, model, network):
        """
        Get the current average value of the parameter of the grouped nodes.
        """
        values = [network.node_attribute(key, name=self.attr_name) for key in self.key]
        return sum(values) / len(values)


class VertexGroupParameter(ParameterGroup):
    """
    A single parameter applied to a group of nodes.
    """
    def index(self, model, structure):
        """
        Get the indices of the keys of the parametrized vertices of a structure.
        """
        return [structure.vertex_index[key] for key in self.key]

    def value(self, model, mesh):
        """
        Get the current average value of the parameter of the grouped vertices.
        """
        values = [mesh.vertex_attribute(key, name=self.attr_name) for key in self.key]
        return sum(values) / len(values)


class EdgeGroupParameter(ParameterGroup):
    """
    A single parameter applied to a group of edges.
    """
    def index(self, model, structure):
        """
        Get the indices of the keys of the parametrized edges from the structure of a model.
        """
        return [structure.edge_index[key] for key in self.key]

    def value(self, model, datastructure):
        """
        Get the current average value of the parameter of the grouped edges.
        """
        values = [datastructure.edge_attribute(key, name=self.attr_name) for key in self.key]
        return sum(values) / len(values)


# ==========================================================================
# Edge force densities
# ==========================================================================

class EdgeForceDensityParameter(EdgeParameter):
    """
    An edge force density parameter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "q"


class EdgeGroupForceDensityParameter(EdgeGroupParameter, EdgeForceDensityParameter):
    """
    A single force density value to rule them all.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================================================
# Node supports
# ==========================================================================

class NodeSupportParameter(NodeParameter):
    """
    A node support parameter.
    """
    def index(self, model, structure):
        """
        Get the index of the key of the node support in the structure of a model.
        """
        return structure.support_index[self.key]


class NodeSupportXParameter(NodeSupportParameter):
    """
    Parametrize the X coordinate of a support node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "x"


class NodeSupportYParameter(NodeSupportParameter):
    """
    Parametrize the Y coordinate of a support node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "y"


class NodeSupportZParameter(NodeSupportParameter):
    """
    Parametrize the Z coordinate of a support node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "z"


# ==========================================================================
# Node supports groups
# ==========================================================================


class NodeGroupSupportParameter(NodeGroupParameter):
    """
    Parametrize a group of support nodes.
    """
    def index(self, model, structure):
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [structure.support_index[key] for key in self.key]


class NodeGroupSupportXParameter(NodeGroupSupportParameter, NodeSupportXParameter):
    """
    Parametrize wiht a single value the X coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodeGroupSupportYParameter(NodeGroupSupportParameter, NodeSupportYParameter):
    """
    Parametrize wiht a single value the Y coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodeGroupSupportZParameter(NodeGroupSupportParameter, NodeSupportZParameter):
    """
    Parametrize wiht a single value the Z coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================================================
# Node loads
# ==========================================================================

class NodeLoadParameter(NodeParameter):
    pass


class NodeLoadXParameter(NodeLoadParameter):
    """
    Parametrize the x component of a node load.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "px"


class NodeLoadYParameter(NodeLoadParameter):
    """
    Parametrize the y component of a node load.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "py"


class NodeLoadZParameter(NodeLoadParameter):
    """
    Parametrize the z component of a node load.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "pz"


# ==========================================================================
# Node loads groups
# ==========================================================================

class NodeGroupLoadXParameter(NodeGroupParameter, NodeLoadXParameter):
    """
    Parametrize wiht a single value the X component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodeGroupLoadYParameter(NodeGroupParameter, NodeLoadXParameter):
    """
    Parametrize wiht a single value the Y component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodeGroupLoadZParameter(NodeGroupParameter, NodeLoadZParameter):
    """
    Parametrize wiht a single value the Z component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex supports
# ==========================================================================

class VertexSupportParameter(VertexParameter):
    """
    A vertex support parameter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def index(self, model, structure):
        """
        Get the index of the key of the vertex support in the structure of a model.
        """
        return structure.support_index[self.key]


class VertexSupportXParameter(VertexSupportParameter, NodeSupportXParameter):
    """
    Parametrize the X coordinate of a support verte.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexSupportYParameter(VertexSupportParameter, NodeSupportYParameter):
    """
    Parametrize the Y coordinate of a support vertex.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexSupportZParameter(VertexSupportParameter, NodeSupportZParameter):
    """
    Parametrize the Z coordinate of a support vertex.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex supports groups
# ==========================================================================


class VertexGroupSupportParameter(NodeGroupParameter):
    """
    Parametrize a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexGroupSupportXParameter(VertexGroupParameter, VertexSupportXParameter):
    """
    Parametrize wiht a single value the X coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexGroupSupportYParameter(VertexGroupParameter, VertexSupportYParameter):
    """
    Parametrize wiht a single value the Y coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexGroupSupportZParameter(VertexGroupParameter, VertexSupportZParameter):
    """
    Parametrize wiht a single value the Z coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex loads
# ==========================================================================

class VertexLoadParameter(VertexParameter):
    pass


class VertexLoadXParameter(VertexLoadParameter, NodeLoadXParameter):
    """
    Parametrize the x component of a node load.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexLoadYParameter(VertexLoadParameter, NodeLoadYParameter):
    """
    Parametrize the y component of a node load.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexLoadZParameter(VertexLoadParameter, NodeLoadZParameter):
    """
    Parametrize the z component of a node load.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex loads groups
# ==========================================================================

class VertexGroupLoadXParameter(VertexGroupParameter, VertexLoadXParameter):
    """
    Parametrize wiht a single value the X component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertesGroupLoadYParameter(VertexGroupParameter, VertexLoadXParameter):
    """
    Parametrize wiht a single value the Y component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VertexGroupLoadZParameter(VertexGroupParameter, VertexLoadZParameter):
    """
    Parametrize wiht a single value the Z component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
