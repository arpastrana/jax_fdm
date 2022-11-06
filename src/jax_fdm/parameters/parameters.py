from jax.numpy import inf


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
        Get the current value of the parameter.
        """
        raise NotImplementedError


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


class EdgeForceDensityParameter(EdgeParameter):
    """
    An edge force density parameter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "q"


class NodeAnchorParameter(NodeParameter):
    """
    A node anchor parameter.
    """
    def index(self, model):
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        return model.structure.anchor_index[self.key]


class NodeLoadParameter(NodeParameter):
    pass


class NodeAnchorXParameter(NodeAnchorParameter):
    """
    Parametrize the z coordinate of an anchor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "x"


class NodeAnchorYParameter(NodeAnchorParameter):
    """
    Parametrize the y coordinate of an anchor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "y"


class NodeAnchorZParameter(NodeAnchorParameter):
    """
    Parametrize the z coordinate of an anchor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_name = "z"


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
