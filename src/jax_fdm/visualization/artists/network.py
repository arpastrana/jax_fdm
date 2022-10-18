from math import fabs

from collections.abc import Iterable

from compas.artists import NetworkArtist

from compas.colors import Color
from compas.colors import ColorMap

from compas.utilities import remap_values


__all__ = ["FDNetworkArtist"]


class FDNetworkArtist(NetworkArtist):
    """
    The base artist to display a force density network across different contexts.
    """
    default_edgecolor = Color.teal()
    default_nodecolor = Color.grey().lightened(factor=100)
    default_nodesupportcolor = Color.from_rgb255(0, 150, 10)

    default_loadcolor = Color.from_rgb255(0, 150, 10)
    default_reactioncolor = Color.pink()

    default_fdcolormap = ColorMap.from_mpl("viridis")
    default_forcecolormap = ColorMap.from_three_colors(Color.from_rgb255(12, 119, 184),
                                                       Color.grey().lightened(50),
                                                       Color.from_rgb255(227, 6, 75))

    default_nodesize = 0.1
    default_edgewidth = (0.01, 0.1)
    default_loadscale = 1.0
    default_reactionscale = 1.0
    default_loadtol = 1e-3
    default_reactiontol = 1e-3

    def __init__(self,
                 network,
                 nodes=None,
                 edges=None,
                 nodecolor=None,
                 edgecolor=None,
                 nodesize=None,
                 edgewidth=None,
                 loadcolor=None,
                 loadscale=None,
                 loadtol=None,
                 reactioncolor=None,
                 reactionscale=None,
                 reactiontol=None,
                 show_nodes=False,
                 show_edges=True,
                 show_loads=True,
                 show_reactions=True,
                 show_supports=True,
                 *args,
                 **kwargs):
        super().__init__(network=network,
                         nodes=nodes,
                         edges=edges,
                         nodecolor=nodecolor,
                         edgecolor=edgecolor,
                         *args,
                         **kwargs)

        self._default_loadcolor = None
        self._default_reactioncolor = None
        self._default_nodesupportcolor = None

        self._default_fdcolormap = None
        self._default_forcecolormap = None

        self._node_size = None

        self.node_size = nodesize
        self.edge_width = edgewidth

        self.load_color = loadcolor or self.default_loadcolor
        self.reaction_color = reactioncolor or self.default_reactioncolor

        self.load_scale = loadscale or self.default_loadscale
        self.load_tol = loadtol or self.default_loadtol
        self.reaction_scale = reactionscale or self.default_reactionscale
        self.reaction_tol = reactiontol or self.default_reactiontol

        self.show_nodes = show_nodes
        self.show_edges = show_edges
        self.show_loads = show_loads
        self.show_reactions = show_reactions
        self.show_supports = show_supports

    # ==========================================================================
    # Draw
    # ==========================================================================

    def draw(self):
        """
        Draw everything.
        """
        data = []

        if self.show_edges:
            edges = self.draw_edges()
            data.extend(edges)
        if self.show_nodes:
            nodes = self.draw_nodes()
            data.extend(nodes)
        if self.show_loads:
            loads = self.draw_loads()
            data.extend(loads)
        if self.show_reactions:
            reactions = self.draw_reactions()
            data.extend(reactions)

        return data

    # ==========================================================================
    # Draw collections
    # ==========================================================================

    def draw_nodes(self):
        """
        Draw the nodes of the network.
        """
        nodes = []
        for node in self.nodes:
            size = self.node_size[node]
            color = self.node_color[node]
            node = self.draw_node(node, size, color)
            nodes.append(node)

        return nodes

    def draw_edges(self):
        """
        Draw the edges of the network.
        """
        edges = []

        for edge in self.edges:
            width = self.edge_width[edge]
            color = self.edge_color[edge]
            edge = self.draw_edge(edge, width, color)
            edges.append(edge)

        return edges

    def draw_loads(self):
        """
        Draw the loads at the nodes of the network.
        """
        loads = []

        for node in self.nodes:
            load = self.draw_load(node, self.load_scale, self.load_color)
            if load:
                loads.append(load)

        return loads

    def draw_reactions(self):
        """
        Draw the reactions at the nodes of the network.
        """
        reactions = []

        for node in self.nodes:
            reaction = self.draw_reaction(node, self.reaction_scale, self.reaction_color)
            if reaction:
                reactions.append(reaction)

        return reactions

    # ==========================================================================
    # Draw elements
    # ==========================================================================

    def draw_node(self, node, size, color):
        """
        Draw a node.
        """
        raise NotImplementedError

    def draw_edge(self, edge, width, color):
        """
        Draw an edge.
        """
        raise NotImplementedError

    def draw_load(self, node, scale, color):
        """
        Draw a load.
        """
        raise NotImplementedError

    def draw_reaction(self, node, scale, color):
        """
        Draw a load.
        """
        raise NotImplementedError

    def clear_nodes(self):
        """
        Clear the nodes.
        """
        pass

    def clear_edges(self):
        """
        Clear the edges.
        """
        pass

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def edge_color(self):
        """
        The edge colors.
        """
        if not self._edge_color:
            self._edge_color = {edge: self.default_edgecolor for edge in self.edges}
        return self._edge_color

    @edge_color.setter
    def edge_color(self, color):
        if isinstance(color, dict):
            self._edge_color = color

        elif isinstance(color, Color):
            self._edge_color = {edge: color for edge in self.edges}

        elif isinstance(color, str):
            network = self.network

            if color == "fd":
                cmap = self.default_fdcolormap
                values = [fabs(network.edge_forcedensity(edge)) for edge in self.edges]

                try:
                    ratios = remap_values(values)
                except ZeroDivisionError:
                    ratios = [0.0] * len(self.edges)

            elif color == "force":
                cmap = self.default_forcecolormap

                forces = []
                compression = []
                tension = []
                for edge in self.edges:
                    force = network.edge_force(edge)
                    forces.append(force)
                    if force <= 0.0:
                        compression.append(force)
                    else:
                        tension.append(force)

                if compression:
                    cmin = min(compression)

                if tension:
                    tmax = max(tension)

                ratios = []
                for force in forces:
                    if force <= 0.0:
                        ratio = remap_values([force], 0.0, 0.5, cmin, 0.0).pop()
                    else:
                        ratio = remap_values([force], 0.5, 1.0, 0.0, tmax).pop()
                    ratios.append(ratio)

            self._edge_color = {edge: cmap(ratio) for edge, ratio in zip(self.edges, ratios)}

    @property
    def node_color(self):
        """
        The node colors.
        """
        if self._node_color:
            return self._node_color

        node_color = {}

        for node in self.nodes:
            color = self.default_nodecolor
            if self.network.node_attribute(node, "is_support"):
                color = self.default_nodesupportcolor
            node_color[node] = color

        self._node_color = node_color

        return self._node_color

    @node_color.setter
    def node_color(self, color):
        if isinstance(color, dict):
            self._node_color = color
        elif isinstance(color, Color):
            self._node_color = {node: color for node in self.nodes}

    @property
    def edge_width(self):
        """
        The width of the edges.
        """
        if not self._edge_width:
            self.edge_width = self.default_edgewidth
        return self._edge_width

    @edge_width.setter
    def edge_width(self, width):
        if isinstance(width, dict):
            self._edge_width = width

        elif isinstance(width, (int, float)):
            self._edge_width = {edge: width for edge in self.edges}

        elif isinstance(width, Iterable) and len(width) == 2:
            width_min, width_max = width
            forces = [fabs(self.network.edge_force(edge)) for edge in self.edges]

            try:
                widths = remap_values(forces, width_min, width_max)
            except ZeroDivisionError:
                widths = [self.default_edgewidth] * len(self.edges)

            self._edge_width = {edge: width for edge, width in zip(self.edges, widths)}
