from typing import Any

from jax.numpy import inf

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure

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
    bound_up : `float`, optional
        The upper bound of this parameter for optimization.
        Defaults to `+inf`.
    """
    def __init__(
        self,
        key: int | tuple[int, ...],
        bound_low: float | None = None,
        bound_up: float | None = None,
    ) -> None:
        """
        Initialize the parameter.
        """
        self.key = key
        self.attr_name: str | None = None

        self._bound_low: float | None = None
        self.bound_low = bound_low

        self._bound_up: float | None = None
        self.bound_up = bound_up

    @property
    def bound_low(self) -> float | None:
        """
        The lower bound of the parameter.
        """
        return self._bound_low

    @property
    def bound_up(self) -> float | None:
        """
        The upper bound of the parameter.
        """
        return self._bound_up

    @bound_low.setter
    def bound_low(self, value: float | None) -> None:
        if value is None:
            value = -inf
        self._bound_low = value

    @bound_up.setter
    def bound_up(self, value: float | None) -> None:
        if value is None:
            value = inf
        self._bound_up = value

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int:
        """
        Get the index of the parameter key in the structure of a model.
        """
        raise NotImplementedError

    def value(self, model: EquilibriumModel, network: FDNetwork | FDMesh) -> float:
        """
        Get the current value of a parameter from the structure of a model.
        """
        raise NotImplementedError


# ==========================================================================
# Individual parameters
# ==========================================================================

class NodeParameter(Parameter):
    """
    A node parameter.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        # a non-group parameter's key is always a bare int at runtime
        return structure.node_index[self.key]  # pyright: ignore[reportArgumentType]

    def value(self, model: EquilibriumModel, network: FDNetwork) -> float:
        """
        Get the current value of the node parameter.
        """
        # key is a bare int; compas returns Any/None but the attribute always holds a float here
        return network.node_attribute(self.key, name=self.attr_name)  # pyright: ignore[reportArgumentType, reportReturnType]


class VertexParameter(Parameter):
    """
    A vertex parameter.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> int:
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        # a non-group parameter's key is always a bare int at runtime
        return structure.vertex_index[self.key]  # pyright: ignore[reportArgumentType]

    def value(self, model: EquilibriumModel, mesh: FDMesh) -> float:
        """
        Get the current value of the node parameter.
        """
        # compas returns Any/None but the attribute always holds a float here
        return mesh.vertex_attribute(self.key, name=self.attr_name)  # pyright: ignore[reportReturnType]


class EdgeParameter(Parameter):
    """
    An edge parameter.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        Get the index of the key of the edge parameter in the structure of a model.
        """
        # a non-group parameter's key is always a bare (u, v) tuple at runtime
        return structure.edge_index[self.key]  # pyright: ignore[reportArgumentType]

    def value(self, model: EquilibriumModel, datastructure: FDNetwork | FDMesh) -> float:
        """
        Get the current value of the edge parameter.
        """
        # key is a bare (u, v) tuple; compas returns Any/None but the attribute always holds a float here
        return datastructure.edge_attribute(self.key, name=self.attr_name)  # pyright: ignore[reportArgumentType, reportReturnType]


# ==========================================================================
# Parameter groups
# ==========================================================================

class ParameterGroup(Parameter):
    """
    A parent class for groups of parameters.
    """
    key: tuple[int, ...]  # a group's key is always a sequence of keys, never a bare int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert len(self.key) > 0

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> list[int]:
        """
        Get the indices of the keys of the parametrized group from the structure of a model.
        """
        raise NotImplementedError


class NodeGroupParameter(ParameterGroup):
    """
    A single parameter applied to a group of nodes.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> list[int]:
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [structure.node_index[key] for key in self.key]

    def value(self, model: EquilibriumModel, network: FDNetwork) -> float:
        """
        Get the current average value of the parameter of the grouped nodes.
        """
        values = [network.node_attribute(key, name=self.attr_name) for key in self.key]
        # compas returns Any/None but every attribute value here is a float
        return sum(values) / len(values)  # pyright: ignore[reportCallIssue, reportArgumentType]


class VertexGroupParameter(ParameterGroup):
    """
    A single parameter applied to a group of nodes.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> list[int]:
        """
        Get the indices of the keys of the parametrized vertices of a structure.
        """
        return [structure.vertex_index[key] for key in self.key]

    def value(self, model: EquilibriumModel, mesh: FDMesh) -> float:
        """
        Get the current average value of the parameter of the grouped vertices.
        """
        values = [mesh.vertex_attribute(key, name=self.attr_name) for key in self.key]
        # compas returns Any/None but every attribute value here is a float
        return sum(values) / len(values)  # pyright: ignore[reportCallIssue, reportArgumentType]


class EdgeGroupParameter(ParameterGroup):
    """
    A single parameter applied to a group of edges.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> list[int]:
        """
        Get the indices of the keys of the parametrized edges from the structure of a model.
        """
        # an edge group key is always a sequence of (u, v) tuples at runtime
        return [structure.edge_index[key] for key in self.key]  # pyright: ignore[reportArgumentType]

    def value(self, model: EquilibriumModel, datastructure: FDNetwork | FDMesh) -> float:
        """
        Get the current average value of the parameter of the grouped edges.
        """
        values = [datastructure.edge_attribute(key, name=self.attr_name) for key in self.key]
        # compas returns Any/None but every attribute value here is a float
        return sum(values) / len(values)  # pyright: ignore[reportCallIssue, reportArgumentType]


# ==========================================================================
# Edge force densities
# ==========================================================================

class EdgeForceDensityParameter(EdgeParameter):
    """
    An edge force density parameter.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "q"


class EdgeGroupForceDensityParameter(EdgeGroupParameter, EdgeForceDensityParameter):
    """
    A single force density value to rule them all.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# ==========================================================================
# Node supports
# ==========================================================================

class NodeSupportParameter(NodeParameter):
    """
    A node support parameter.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        Get the index of the key of the node support in the structure of a model.
        """
        # a non-group parameter's key is always a bare int at runtime
        return structure.support_index[self.key]  # pyright: ignore[reportArgumentType]


class NodeSupportXParameter(NodeSupportParameter):
    """
    Parametrize the X coordinate of a support node.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "x"


class NodeSupportYParameter(NodeSupportParameter):
    """
    Parametrize the Y coordinate of a support node.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "y"


class NodeSupportZParameter(NodeSupportParameter):
    """
    Parametrize the Z coordinate of a support node.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "z"


# ==========================================================================
# Node supports groups
# ==========================================================================


class NodeGroupSupportParameter(NodeGroupParameter):
    """
    Parametrize a group of support nodes.
    """
    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> list[int]:
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [structure.support_index[key] for key in self.key]


class NodeGroupSupportXParameter(NodeGroupSupportParameter, NodeSupportXParameter):
    """
    Parametrize with a single value the X coordinate of a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class NodeGroupSupportYParameter(NodeGroupSupportParameter, NodeSupportYParameter):
    """
    Parametrize with a single value the Y coordinate of a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class NodeGroupSupportZParameter(NodeGroupSupportParameter, NodeSupportZParameter):
    """
    Parametrize with a single value the Z coordinate of a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "px"


class NodeLoadYParameter(NodeLoadParameter):
    """
    Parametrize the y component of a node load.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "py"


class NodeLoadZParameter(NodeLoadParameter):
    """
    Parametrize the z component of a node load.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attr_name = "pz"


# ==========================================================================
# Node loads groups
# ==========================================================================

class NodeGroupLoadXParameter(NodeGroupParameter, NodeLoadXParameter):
    """
    Parametrize with a single value the X component of the load applied to a group of nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class NodeGroupLoadYParameter(NodeGroupParameter, NodeLoadYParameter):
    """
    Parametrize with a single value the Y component of the load applied to a group of nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class NodeGroupLoadZParameter(NodeGroupParameter, NodeLoadZParameter):
    """
    Parametrize with a single value the Z component of the load applied to a group of nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex supports
# ==========================================================================

class VertexSupportParameter(VertexParameter):
    """
    A vertex support parameter.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        Get the index of the key of the vertex support in the structure of a model.
        """
        return structure.support_index[self.key]  # pyright: ignore[reportArgumentType]  # a non-group parameter's key is always a bare int at runtime


class VertexSupportXParameter(VertexSupportParameter, NodeSupportXParameter):
    """
    Parametrize the X coordinate of a support verte.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexSupportYParameter(VertexSupportParameter, NodeSupportYParameter):
    """
    Parametrize the Y coordinate of a support vertex.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexSupportZParameter(VertexSupportParameter, NodeSupportZParameter):
    """
    Parametrize the Z coordinate of a support vertex.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex supports groups
# ==========================================================================


class VertexGroupSupportParameter(VertexGroupParameter):
    """
    Parametrize a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> list[int]:
        """
        Get the indices of the keys of the parametrized vertices of the structure of a model.
        """
        return [structure.support_index[key] for key in self.key]


class VertexGroupSupportXParameter(VertexGroupSupportParameter, VertexSupportXParameter):
    """
    Parametrize with a single value the X coordinate of a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexGroupSupportYParameter(VertexGroupSupportParameter, VertexSupportYParameter):
    """
    Parametrize with a single value the Y coordinate of a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexGroupSupportZParameter(VertexGroupSupportParameter, VertexSupportZParameter):
    """
    Parametrize with a single value the Z coordinate of a group of support nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexLoadYParameter(VertexLoadParameter, NodeLoadYParameter):
    """
    Parametrize the y component of a node load.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexLoadZParameter(VertexLoadParameter, NodeLoadZParameter):
    """
    Parametrize the z component of a node load.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# ==========================================================================
# Vertex loads groups
# ==========================================================================

class VertexGroupLoadXParameter(VertexGroupParameter, VertexLoadXParameter):
    """
    Parametrize with a single value the X component of the load applied to a group of nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexGroupLoadYParameter(VertexGroupParameter, VertexLoadYParameter):
    """
    Parametrize with a single value the Y component of the load applied to a group of nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class VertexGroupLoadZParameter(VertexGroupParameter, VertexLoadZParameter):
    """
    Parametrize with a single value the Z component of the load applied to a group of nodes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
