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
    The base class for all optimization parameters.

    Parameters
    ----------
    key :
        The key of the element in the datastructure being parametrized.
    bound_low :
        The lower bound for optimization. If None, unbounded below.
    bound_up :
        The upper bound for optimization. If None, unbounded above.

    Notes
    -----
    Concrete subclasses set ``attr_name`` to the datastructure attribute they
    parametrize and implement :meth:`index` and :meth:`value` for their element type.
    Missing bounds normalize to negative or positive infinity rather than None.
    """

    attr_name: str | None = (
        None  # the datastructure attribute name; set per concrete subclass
    )
    key: (
        int | tuple[int, int] | list[int] | list[tuple[int, int]]
    )  # narrowed per concrete subclass

    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        bound_low: float | None = None,
        bound_up: float | None = None,
    ) -> None:
        """
        Initialize the parameter.
        """
        self.key = key

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
        Resolve the parameter's key to an index in an equilibrium structure.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            The index of the parametrized element.
        """
        raise NotImplementedError

    def value(
        self,
        model: EquilibriumModel,
        datastructure: FDNetwork | FDMesh,
    ) -> float:
        """
        Read the parameter's current value from a datastructure.

        Parameters
        ----------
        model :
            The equilibrium model.
        datastructure :
            The network or mesh to read the parametrized attribute from.

        Returns
        -------
        value :
            The current value of the parametrized attribute.
        """
        raise NotImplementedError


# ==========================================================================
# Individual parameters
# ==========================================================================


class NodeParameter(Parameter):
    """
    A parameter on a single network node.
    """

    key: int  # a non-group node key is always a bare int at runtime

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        The index of the parametrized node in the structure.
        """
        return structure.node_index[self.key]

    def value(self, model: EquilibriumModel, datastructure: FDNetwork) -> float:
        """
        The current value of the parametrized node attribute.
        """
        # compas accessors are untyped but the attribute always holds a float here
        return datastructure.node_attribute(self.key, name=self.attr_name)  # pyright: ignore[reportReturnType]


class VertexParameter(Parameter):
    """
    A parameter on a single mesh vertex.
    """

    key: int  # a non-group vertex key is always a bare int at runtime

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> int:
        """
        The index of the parametrized vertex in the structure.
        """
        return structure.vertex_index[self.key]

    def value(self, model: EquilibriumModel, datastructure: FDMesh) -> float:
        """
        The current value of the parametrized vertex attribute.
        """
        # compas accessors are untyped but the attribute always holds a float here
        return datastructure.vertex_attribute(self.key, name=self.attr_name)  # pyright: ignore[reportReturnType]


class EdgeParameter(Parameter):
    """
    A parameter on a single network edge.
    """

    # a non-group edge key is always a bare (u, v) tuple at runtime
    key: tuple[int, int]

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        The index of the parametrized edge in the structure.
        """
        return structure.edge_index[self.key]

    def value(
        self,
        model: EquilibriumModel,
        datastructure: FDNetwork | FDMesh,
    ) -> float:
        """
        The current value of the parametrized edge attribute.
        """
        # compas accessors are untyped but the attribute always holds a float here
        return datastructure.edge_attribute(self.key, name=self.attr_name)  # pyright: ignore[reportReturnType]


# ==========================================================================
# Parameter groups
# ==========================================================================


class ParameterGroup(Parameter):
    """
    The base class for a single parameter shared across a group of elements.

    Notes
    -----
    A group parameter drives one value that is applied to, and read as the average
    over, all its elements. Its key is always a sequence of element keys.
    """

    # a group's key is always a sequence of keys, never a bare int
    key: tuple[int, ...]

    def __init__(
        self,
        key: list[int] | list[tuple[int, int]],
        bound_low: float | None = None,
        bound_up: float | None = None,
    ) -> None:
        """
        Initialize the parameter group.

        Parameters
        ----------
        key :
            The keys of the elements sharing the parameter.
        bound_low :
            The lower bound for optimization. If None, unbounded below.
        bound_up :
            The upper bound for optimization. If None, unbounded above.
        """
        super().__init__(key, bound_low, bound_up)
        assert len(self.key) > 0

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> list[int]:
        """
        Resolve the group's keys to indices in an equilibrium structure.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose element ordering defines the indices.

        Returns
        -------
        indices :
            The indices of the parametrized elements.
        """
        raise NotImplementedError


class NodeGroupParameter(ParameterGroup):
    """
    A single parameter shared across a group of network nodes.
    """

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> list[int]:
        """
        The indices of the parametrized nodes in the structure.
        """
        return [structure.node_index[key] for key in self.key]

    def value(self, model: EquilibriumModel, datastructure: FDNetwork) -> float:
        """
        The current mean value of the parametrized attribute over the grouped nodes.
        """
        values = [
            datastructure.node_attribute(key, name=self.attr_name) for key in self.key
        ]
        # compas accessors are untyped but every attribute value here is a float
        return sum(values) / len(values)  # pyright: ignore[reportCallIssue, reportArgumentType]


class VertexGroupParameter(ParameterGroup):
    """
    A single parameter shared across a group of mesh vertices.
    """

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> list[int]:
        """
        The indices of the parametrized vertices in the structure.
        """
        return [structure.vertex_index[key] for key in self.key]

    def value(self, model: EquilibriumModel, datastructure: FDMesh) -> float:
        """
        The current mean value of the parametrized attribute over the grouped vertices.
        """
        values = [
            datastructure.vertex_attribute(key, name=self.attr_name) for key in self.key
        ]
        # compas accessors are untyped but every attribute value here is a float
        return sum(values) / len(values)  # pyright: ignore[reportCallIssue, reportArgumentType]


class EdgeGroupParameter(ParameterGroup):
    """
    A single parameter shared across a group of network edges.
    """

    # an edge group key is always a sequence of (u, v) tuples at runtime
    key: tuple[tuple[int, int], ...]

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> list[int]:
        """
        The indices of the parametrized edges in the structure.
        """
        return [structure.edge_index[key] for key in self.key]

    def value(
        self,
        model: EquilibriumModel,
        datastructure: FDNetwork | FDMesh,
    ) -> float:
        """
        The current mean value of the parametrized attribute over the grouped edges.
        """
        values = [
            datastructure.edge_attribute(key, name=self.attr_name) for key in self.key
        ]
        # compas accessors are untyped but every attribute value here is a float
        return sum(values) / len(values)  # pyright: ignore[reportCallIssue, reportArgumentType]


# ==========================================================================
# Edge force densities
# ==========================================================================


class EdgeForceDensityParameter(EdgeParameter):
    """
    An edge force density parameter.
    """

    attr_name = "q"


class EdgeGroupForceDensityParameter(EdgeGroupParameter, EdgeForceDensityParameter):
    """
    A single force density value to rule them all: one value shared across a
    group of edges.
    """


# ==========================================================================
# Node supports
# ==========================================================================


class NodeSupportParameter(NodeParameter):
    """
    A node support parameter.
    """

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        The index of the parametrized node among the structure's supports.
        """
        return structure.support_index[self.key]


class NodeSupportXParameter(NodeSupportParameter):
    """
    Parametrize the X coordinate of a support node.
    """

    attr_name = "x"


class NodeSupportYParameter(NodeSupportParameter):
    """
    Parametrize the Y coordinate of a support node.
    """

    attr_name = "y"


class NodeSupportZParameter(NodeSupportParameter):
    """
    Parametrize the Z coordinate of a support node.
    """

    attr_name = "z"


# ==========================================================================
# Node supports groups
# ==========================================================================


class NodeGroupSupportParameter(NodeGroupParameter):
    """
    Parametrize a group of support nodes.
    """

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> list[int]:
        """
        The indices of the parametrized nodes among the structure's supports.
        """
        return [structure.support_index[key] for key in self.key]


class NodeGroupSupportXParameter(NodeGroupSupportParameter, NodeSupportXParameter):
    """
    Parametrize with a single value the X coordinate of a group of support nodes.
    """


class NodeGroupSupportYParameter(NodeGroupSupportParameter, NodeSupportYParameter):
    """
    Parametrize with a single value the Y coordinate of a group of support nodes.
    """


class NodeGroupSupportZParameter(NodeGroupSupportParameter, NodeSupportZParameter):
    """
    Parametrize with a single value the Z coordinate of a group of support nodes.
    """


# ==========================================================================
# Node loads
# ==========================================================================


class NodeLoadParameter(NodeParameter):
    """
    A node load parameter.
    """


class NodeLoadXParameter(NodeLoadParameter):
    """
    Parametrize the x component of a node load.
    """

    attr_name = "px"


class NodeLoadYParameter(NodeLoadParameter):
    """
    Parametrize the y component of a node load.
    """

    attr_name = "py"


class NodeLoadZParameter(NodeLoadParameter):
    """
    Parametrize the z component of a node load.
    """

    attr_name = "pz"


# ==========================================================================
# Node loads groups
# ==========================================================================


class NodeGroupLoadXParameter(NodeGroupParameter, NodeLoadXParameter):
    """
    Parametrize with a single value the X component of the load applied to a group of
    nodes.
    """


class NodeGroupLoadYParameter(NodeGroupParameter, NodeLoadYParameter):
    """
    Parametrize with a single value the Y component of the load applied to a group of
    nodes.
    """


class NodeGroupLoadZParameter(NodeGroupParameter, NodeLoadZParameter):
    """
    Parametrize with a single value the Z component of the load applied to a group of
    nodes.
    """


# ==========================================================================
# Vertex supports
# ==========================================================================


class VertexSupportParameter(VertexParameter):
    """
    A vertex support parameter.
    """

    def index(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        The index of the parametrized vertex among the structure's supports.
        """
        return structure.support_index[self.key]


class VertexSupportXParameter(VertexSupportParameter, NodeSupportXParameter):
    """
    Parametrize the X coordinate of a support vertex.
    """


class VertexSupportYParameter(VertexSupportParameter, NodeSupportYParameter):
    """
    Parametrize the Y coordinate of a support vertex.
    """


class VertexSupportZParameter(VertexSupportParameter, NodeSupportZParameter):
    """
    Parametrize the Z coordinate of a support vertex.
    """


# ==========================================================================
# Vertex supports groups
# ==========================================================================


class VertexGroupSupportParameter(VertexGroupParameter):
    """
    Parametrize a group of support vertices.
    """

    def index(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> list[int]:
        """
        The indices of the parametrized vertices among the structure's supports.
        """
        return [structure.support_index[key] for key in self.key]


class VertexGroupSupportXParameter(
    VertexGroupSupportParameter,
    VertexSupportXParameter,
):
    """
    Parametrize with a single value the X coordinate of a group of support vertices.
    """


class VertexGroupSupportYParameter(
    VertexGroupSupportParameter,
    VertexSupportYParameter,
):
    """
    Parametrize with a single value the Y coordinate of a group of support vertices.
    """


class VertexGroupSupportZParameter(
    VertexGroupSupportParameter,
    VertexSupportZParameter,
):
    """
    Parametrize with a single value the Z coordinate of a group of support vertices.
    """


# ==========================================================================
# Vertex loads
# ==========================================================================


class VertexLoadXParameter(VertexParameter, NodeLoadXParameter):
    """
    Parametrize the x component of a vertex load.
    """


class VertexLoadYParameter(VertexParameter, NodeLoadYParameter):
    """
    Parametrize the y component of a vertex load.
    """


class VertexLoadZParameter(VertexParameter, NodeLoadZParameter):
    """
    Parametrize the z component of a vertex load.
    """


# ==========================================================================
# Vertex loads groups
# ==========================================================================


class VertexGroupLoadXParameter(VertexGroupParameter, VertexLoadXParameter):
    """
    Parametrize with a single value the X component of the load applied to a group of
    vertices.
    """


class VertexGroupLoadYParameter(VertexGroupParameter, VertexLoadYParameter):
    """
    Parametrize with a single value the Y component of the load applied to a group of
    vertices.
    """


class VertexGroupLoadZParameter(VertexGroupParameter, VertexLoadZParameter):
    """
    Parametrize with a single value the Z component of the load applied to a group of
    vertices.
    """
