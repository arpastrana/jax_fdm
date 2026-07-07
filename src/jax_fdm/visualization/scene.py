from math import fabs
from typing import Callable
from typing import NamedTuple

import numpy as np

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from compas.itertools import remap_values

__all__ = [
    "PointAccess",
    "network_accessors",
    "mesh_accessors",
    "edge_colors",
    "edge_widths",
    "point_colors",
    "point_sizes",
    "load_arrows",
    "reaction_arrows",
    "reaction_color_default",
]


# ==========================================================================
# Defaults
# ==========================================================================

# Colors
COLOR_EDGE = Color.teal()
COLOR_POINT = Color.grey().lightened(factor=50)
COLOR_SUPPORT = Color.from_rgb255(0, 150, 10)

COLOR_LOAD = Color.from_rgb255(0, 150, 10)
COLOR_REACTION = Color.pink()

COLOR_TENSION = Color.from_rgb255(227, 6, 75)
COLOR_COMPRESSION = Color.from_rgb255(12, 119, 184)

COLORMAP_FD = ColorMap.from_mpl("viridis")

# Sizes and scales
POINT_SIZE = 0.1
EDGE_WIDTH = (0.01, 0.1)
LOAD_SCALE = 1.0
REACTION_SCALE = 1.0
LOAD_TOL = 1e-3
REACTION_TOL = 1e-3

# Arrow proportions, as fractions of the arrow length
ARROW_HEADPORTION = 0.12
ARROW_HEADWIDTH = 0.04
ARROW_BODYWIDTH = 0.012


# ==========================================================================
# Point accessors (node-vs-vertex vocabulary)
# ==========================================================================

class PointAccess(NamedTuple):
    """
    The point accessors of a force density datastructure.

    A network addresses its points as nodes and a mesh as vertices; this
    bundle resolves that vocabulary once so the display functions stay
    datastructure-agnostic. The edge accessors (``edge_coordinates``,
    ``edge_force``, ``edge_forcedensity``) already live on the shared
    :class:`jax_fdm.datastructures.FDDatastructure` and are used directly.
    """
    keys: Callable
    coordinates: Callable
    load: Callable
    reaction: Callable
    edges: Callable
    is_support: Callable


def network_accessors(network):
    """
    The point accessors of a force density network (nodes).
    """
    return PointAccess(keys=network.nodes,
                       coordinates=network.node_coordinates,
                       load=network.node_load,
                       reaction=network.node_reaction,
                       edges=network.node_edges,
                       is_support=lambda key: network.node_attribute(key, "is_support"))


def mesh_accessors(mesh):
    """
    The point accessors of a force density mesh (vertices).
    """
    return PointAccess(keys=mesh.vertices,
                       coordinates=mesh.vertex_coordinates,
                       load=mesh.vertex_load,
                       reaction=mesh.vertex_reaction,
                       edges=mesh.vertex_edges,
                       is_support=lambda key: mesh.vertex_attribute(key, "is_support"))


# ==========================================================================
# Edge styling
# ==========================================================================

def edge_colors(datastructure, edges, color=None):
    """
    Map every edge to a color.

    Parameters
    ----------
    color : :class:`compas.colors.Color` | dict | str, optional
        A single color to broadcast, a per-edge dict, or a semantic mode:
        ``"fd"`` colors by absolute force density through a colormap, and
        ``"force"`` colors tension red and compression blue.
        Defaults to teal.
    """
    if isinstance(color, dict):
        return color

    if isinstance(color, Color):
        return {edge: color for edge in edges}

    if color == "fd":
        cmap = COLORMAP_FD
        values = [fabs(datastructure.edge_forcedensity(edge)) for edge in edges]
        try:
            ratios = remap_values(values)
        except ZeroDivisionError:
            ratios = [0.0] * len(edges)
        return {edge: cmap(ratio) for edge, ratio in zip(edges, ratios)}

    if color == "force":
        return {edge: COLOR_TENSION if datastructure.edge_force(edge) > 0.0 else COLOR_COMPRESSION
                for edge in edges}

    return {edge: COLOR_EDGE for edge in edges}


def edge_widths(datastructure, edges, width=None):
    """
    Map every edge to a width.

    Parameters
    ----------
    width : float | dict | tuple, optional
        A single width to broadcast, a per-edge dict, or a ``(min, max)``
        pair to remap the absolute edge forces into.
        Defaults to the ``(min, max)`` remap with the default bounds.
    """
    if isinstance(width, dict):
        return width

    if isinstance(width, (int, float)):
        return {edge: width for edge in edges}

    if width is None:
        width = EDGE_WIDTH

    if isinstance(width, (tuple, list)) and len(width) == 2:
        width_min, width_max = width

        if not edges:
            return {}

        forces = [fabs(datastructure.edge_force(edge)) for edge in edges]

        if min(forces) == max(forces):
            widths = [width_max] * len(edges)
        else:
            try:
                widths = remap_values(forces, width_min, width_max)
            except ZeroDivisionError:
                widths = [width_max] * len(edges)

        return {edge: width for edge, width in zip(edges, widths)}

    raise ValueError(f"Unsupported edge width: {width}")


# ==========================================================================
# Point styling
# ==========================================================================

def point_colors(points, is_support, color=None):
    """
    Map every point to a color, defaulting supports to green.

    Parameters
    ----------
    is_support : callable
        The support predicate of the datastructure (from :class:`PointAccess`).
    color : :class:`compas.colors.Color` | dict, optional
        A single color to broadcast or a per-point dict.
    """
    if isinstance(color, dict):
        return color

    if isinstance(color, Color):
        return {point: color for point in points}

    return {point: COLOR_SUPPORT if is_support(point) else COLOR_POINT for point in points}


def point_sizes(points, size=None):
    """
    Map every point to a size.

    Parameters
    ----------
    size : float | dict, optional
        A single size to broadcast or a per-point dict.
    """
    if isinstance(size, dict):
        return size

    return {point: size if size is not None else POINT_SIZE for point in points}


def reaction_color_default(edgecolor):
    """
    The default reaction color, given the edge color mode.

    When edges are colored by force, the reactions turn green so they read
    apart from the red-blue tension-compression palette.
    """
    if edgecolor == "force":
        return Color.from_rgb255(0, 150, 10)
    return COLOR_REACTION


# ==========================================================================
# Arrows
# ==========================================================================

def load_arrows(points, access, widths, scale, tol):
    """
    Compute the anchor and vector of the load arrow at every point.

    The arrow is placed so its head touches the point it loads: the anchor
    backs away from the point by the scaled vector, plus the widest connected
    edge so the head clears the edge cylinders.

    Points whose load is below tolerance get a degenerate slot (anchor at the
    point, zero vector) so the arrow count stays constant across updates.

    Returns
    -------
    (anchors, vectors)
        Arrays of shape (N, 3), aligned with ``points``.
    """
    anchors, vectors = [], []

    for point in points:
        xyz = access.coordinates(point)
        vector = access.load(point)

        if length_vector(vector) < tol:
            anchors.append(xyz)
            vectors.append((0.0, 0.0, 0.0))
            continue

        # shift start to make the arrow head touch the loaded point,
        # then back off by the thickest connected edge so the head clears it
        start = add_vectors(xyz, scale_vector(vector, -scale))
        width = max((widths.get(edge, 0.0) for edge in access.edges(point)), default=0.0)
        start = add_vectors(start, scale_vector(normalize_vector(vector), -width))

        anchors.append(start)
        vectors.append(scale_vector(vector, scale))

    return np.asarray(anchors, dtype=np.float64), np.asarray(vectors, dtype=np.float64)


def reaction_arrows(points, access, datastructure, scale, tol):
    """
    Compute the anchor and vector of the reaction arrow at every point.

    The vector is reversed to display the direction the reaction pushes
    against the structure; when the strongest connected edge is compressive,
    the anchor shifts outward so the arrow points at the support from outside.

    Points whose reaction is below tolerance (or without connected edges) get
    a degenerate slot so the arrow count stays constant across updates.

    Returns
    -------
    (anchors, vectors)
        Arrays of shape (N, 3), aligned with ``points``.
    """
    anchors, vectors = [], []

    for point in points:
        xyz = access.coordinates(point)
        vector = access.reaction(point)
        edges = list(access.edges(point))

        if length_vector(vector) < tol or not edges:
            anchors.append(xyz)
            vectors.append((0.0, 0.0, 0.0))
            continue

        start = xyz
        forces = [datastructure.edge_force(edge) for edge in edges]
        max_force = max(forces, key=fabs)
        if max_force < 0.0:
            start = add_vectors(start, scale_vector(vector, scale))

        anchors.append(start)
        vectors.append(scale_vector(vector, -scale))

    return np.asarray(anchors, dtype=np.float64), np.asarray(vectors, dtype=np.float64)
