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


class EdgeParameter(Parameter):
    """
    An edge parameter.
    """
    def index(self, model):
        """
        Get the index of the key of the edge parameter in the structure of a model.
        """
        return model.structure.edge_index[self.key]


# ==========================================================================
# Groups
# ==========================================================================

class ParameterGroup(Parameter):
    """
    from jax_fdm.parameters import ParameterGroup
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodesParameter(ParameterGroup):
    """
    A single parameter applied to a group of nodes.
    """
    def index(self, model):
        """
        Get the indices of the keys of the parametrized nodes from the structure of a model.
        """
        return [model.structure.node_index[key] for key in self.key]


class EdgesParameter(ParameterGroup):
    """
    A single parameter applied to a group of edges.
    """
    def index(self, model):
        """
        Get the indices of the keys of the parametrized edges from the structure of a model.
        """
        return [model.structure.edge_index[key] for key in self.key]

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

class NodeAnchorParameter(NodeParameter):
    """
    A node anchor parameter.
    """
    def index(self, model):
        """
        Get the index of the key of the node parameter in the structure of a model.
        """
        return model.structure.anchor_index[self.key]


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
# Example
# ==========================================================================

if __name__ == "__main__":

    # randomness
    from random import random

    # compas
    from compas.geometry import add_vectors
    from compas.colors import Color
    from compas.geometry import Polyline
    from compas.geometry import Line
    from compas.geometry import length_vector

    # jax fdm
    from jax_fdm.datastructures import FDNetwork
    from jax_fdm.visualization import Plotter

    from jax_fdm.equilibrium import fdm
    from jax_fdm.equilibrium import constrained_fdm
    from jax_fdm.equilibrium import EquilibriumModel

    from jax_fdm.goals import NodeYCoordinateGoal

    from jax_fdm.parameters import ParameterManager
    from jax_fdm.parameters import EdgeForceDensityParameter
    from jax_fdm.parameters import EdgesForceDensityParameter

    from jax_fdm.losses import Loss
    from jax_fdm.losses import SquaredError

    from jax_fdm.optimization import SLSQP


    def create_polyline(length: float, num_segments: int):
        """Create a network that represents a 2D arch.
        """
        start = [-length / 2.0, 0.0, 0.0]
        end = add_vectors(start, [length, 0.0, 0.0])
        curve = Polyline([start, end])
        points = curve.divide_polyline(num_segments)
        for point in points[1:-1]:
            point[1] = random() * 0.001
        return Polyline(points).lines

    def plot_arch(arch, draw_nodelabels=False, *args, **kwargs):
        """Display a funky arch in a plotter.
        """
        plotter = Plotter(figsize=(8, 5), dpi=150)
        artist = plotter.add(arch, show_nodes=True, edgewidth=(0.5, 3.0), loadscale=1.0, nodesize=0.5, *args, **kwargs)
        if draw_nodelabels:
            artist.draw_nodelabels("key")
        plotter.zoom_extents()
        return plotter


    def connect_arches(arch_1, arch_2):
        """Connect the nodes of two arches with a line.
        """
        lines = []
        for na, nb in zip(arch_1.nodes(), arch_2.nodes()):
            line = Line(arch_1.node_coordinates(na), arch_2.node_coordinates(nb))
            lines.append(line)
        return lines


    def arch_height(arch):
        """Compute the maximum y coordinate of the nodes of the arch.
        """
        return max(arch.nodes_attribute(name="y"))


    def plot_connected_arches(arch_1, arch_2):
        """Plot two arches. Their nodes are pairwise connected by a line.
        The second arch is drawn as a polyline (no forces or reactions shown).
        """
        plotter = plot_arch(arch_1)
        plotter.add(arch_2, show_loads=False)
        lines = connect_arches(arch_1, arch_2)
        for line in lines:
            plotter.add(line)
        return plotter

    def plot_horizontal_line(height, plotter):
        """Plot a horizontal line at a given height.
        """
        line_horizontal = Line([0.0, height, 0.0], [1.0, height, 0.0])
        plotter.add(line_horizontal, color=Color.orange())
        return plotter


    # Create a arch modeled as a JAX FDM network
    polyline = create_polyline(length=5.0, num_segments=10)
    arch = FDNetwork.from_lines(polyline)

    # Assign supports to the arch
    arch.node_support(0)
    arch.node_support(10)

    # Apply loads at the nodes
    for node in arch.nodes_free():
        arch.node_load(node, [0.0, -0.5, 0.0])

    # Set edge force densities
    for i, edge in enumerate(arch.edges()):
        arch.edge_forcedensity(edge, -i)

    arch_eq = fdm(arch)
    # arch_eq.print_stats()

    # formulate goals
    goals = []
    goal = NodeYCoordinateGoal(key=5, target=1.0)  # target height
    goals.append(goal)

    # set optimizable parameters
    parameters = []
    for edge in list(arch.edges())[3:4]:
        parameter = EdgeForceDensityParameter(edge, bound_low=-20.0, bound_up=0.0)
        parameters.append(parameter)

    edges = list(arch.edges())[4:6]
    parameter = EdgesForceDensityParameter(edges, bound_low=-10.0, bound_up=-1.0)
    parameters.append(parameter)


    model = EquilibriumModel.from_network(arch)
    manager = ParameterManager(model, parameters)

    print("opt indices")
    print(manager.indices_opt)

    print("bound low parameters")
    print(manager.bounds_low)

    print("opt parameters")
    x = manager.parameters_opt
    print(x)


    # print("fdm parameters reconstructed")
    # params_fdm = manager.parameters_fdm(x)
    # print(params_fdm)

    raise

    # define loss (objective) function
    # error = SquaredError(goals)
    # loss = Loss(error)

    # # instantiate
    # optimizer = SLSQP()

    # # solve constrained form finding problem
    # arch_c = constrained_fdm(arch, optimizer, loss, parameters)
    # arch_c.print_stats()

    # # plot arch
    # plotter = plot_connected_arches(arch_c, arch)
    # plot_horizontal_line(1.0, plotter)
    # plotter.show()
