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

    def value(self, model):
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
    def index(self, model):
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        return model.structure.node_index[self.key]

    def value(self, model):
        """
        Get the current value of the node parameter.
        """
        return model.structure.network.node_attribute(self.key, name=self.attr_name)


class EdgeParameter(Parameter):
    """
    An edge parameter.
    """
    def index(self, model):
        """
        Get the index of the key of the edge parameter in the structure of a model.
        """
        return model.structure.edge_index[self.key]

    def value(self, model):
        """
        Get the current value of the edge parameter.
        """
        return model.structure.network.edge_attribute(self.key, name=self.attr_name)


# ==========================================================================
# ParameterGroups
# ==========================================================================

class ParameterGroup(Parameter):
    """
    A parent class for groups of parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.key) > 0


class NodesParameter(ParameterGroup):
    """
    A single parameter applied to a group of nodes.
    """
    def index(self, model):
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [model.structure.node_index[key] for key in self.key]

    def value(self, model):
        """
        Get the current average value of the parameter of the grouped nodes.
        """
        values = [model.structure.network.node_attribute(key, name=self.attr_name) for key in self.key]
        return sum(values) / len(values)


class EdgesParameter(ParameterGroup):
    """
    A single parameter applied to a group of edges.
    """
    def index(self, model):
        """
        Get the indices of the keys of the parametrized edges from the structure of a model.
        """
        return [model.structure.edge_index[key] for key in self.key]

    def value(self, model):
        """
        Get the current average value of the parameter of the grouped edges.
        """
        values = [model.structure.network.edge_attribute(key, name=self.attr_name) for key in self.key]
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


class EdgesForceDensityParameter(EdgesParameter, EdgeForceDensityParameter):
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
    def index(self, model):
        """
        Get the index of the key of the node support in the structure of a model.
        """
        return model.structure.anchor_index[self.key]


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


class NodesSupportParameter(NodesParameter):
    """
    Parametrize a group of support nodes.
    """
    def index(self, model):
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [model.structure.anchor_index[key] for key in self.key]


class NodesSupportXParameter(NodesSupportParameter, NodeSupportXParameter):
    """
    Parametrize wiht a single value the X coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodesSupportYParameter(NodesSupportParameter, NodeSupportYParameter):
    """
    Parametrize wiht a single value the Y coordinate of a group of support nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodesSupportZParameter(NodesSupportParameter, NodeSupportZParameter):
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

class NodesLoadXParameter(NodesParameter, NodeLoadXParameter):
    """
    Parametrize wiht a single value the X component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodesLoadYParameter(NodesParameter, NodeLoadXParameter):
    """
    Parametrize wiht a single value the Y component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodesLoadZParameter(NodesParameter, NodeLoadZParameter):
    """
    Parametrize wiht a single value the Z component of the load applied to a group of nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
