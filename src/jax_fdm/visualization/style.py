from collections.abc import Callable
from math import fabs

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from compas.itertools import remap_values
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork

__all__ = [
    "edge_colors",
    "edge_widths",
    "point_colors",
    "point_sizes",
    "load_arrows",
    "reaction_arrows",
    "reaction_color_default",
    "EdgeColors",
    "EdgeWidths",
    "PointColors",
    "PointSizes",
    "EdgeColorSpec",
    "EdgeWidthSpec",
    "PointColorSpec",
    "PointSizeSpec",
]

# ==========================================================================
# Type aliases
# ==========================================================================

# Per-element style maps: an edge is keyed by its (u, v) node pair, a point (a
# node or a vertex) by a single key. These are the shapes the style functions
# return and the scene objects store.
EdgeColors = dict[tuple[int, int], Color]
EdgeWidths = dict[tuple[int, int], float]
PointColors = dict[int, Color]
PointSizes = dict[int, float]

# User-facing style specs: a single value broadcast to every element, a
# per-element map, a semantic string mode (e.g. "fd"/"force" for edges), an
# edge-width (low, high) range, or None to fall back to the defaults.
EdgeColorSpec = Color | EdgeColors | str | None
EdgeWidthSpec = float | EdgeWidths | tuple[float, float] | None
PointColorSpec = Color | PointColors | str | None
PointSizeSpec = float | PointSizes | None


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
# Edge styling
# ==========================================================================


def edge_colors(
    datastructure: FDNetwork | FDMesh,
    edges: list[tuple[int, int]],
    color: EdgeColorSpec = None,
) -> EdgeColors:
    """
    Map every edge to a color.

    Parameters
    ----------
    datastructure :
        The network or mesh the edges belong to.
    edges :
        The edges to color.
    color :
        A single color to broadcast, a per-edge dict, or a semantic mode: ``"fd"``
        colors by absolute force density through a colormap, and ``"force"`` colors
        tension red and compression blue. If None, defaults to teal.

    Returns
    -------
    colors :
        A color for each edge.
    """
    if isinstance(color, dict):
        return color

    if isinstance(color, Color):
        return {edge: color for edge in edges}

    if color == "fd":
        cmap = COLORMAP_FD
        # the getter-mode call (no q kwarg) always returns a float
        values = [fabs(datastructure.edge_forcedensity(edge)) for edge in edges]  # pyright: ignore[reportArgumentType]
        try:
            ratios = remap_values(values)
        except ZeroDivisionError:
            ratios = [0.0] * len(edges)
        return {edge: cmap(ratio) for edge, ratio in zip(edges, ratios)}

    if color == "force":
        return {
            edge: COLOR_TENSION
            if datastructure.edge_force(edge) > 0.0
            else COLOR_COMPRESSION
            for edge in edges
        }

    return {edge: COLOR_EDGE for edge in edges}


def edge_widths(
    datastructure: FDNetwork | FDMesh,
    edges: list[tuple[int, int]],
    width: EdgeWidthSpec = None,
) -> EdgeWidths:
    """
    Map every edge to a width.

    Parameters
    ----------
    datastructure :
        The network or mesh the edges belong to.
    edges :
        The edges to size.
    width :
        A single width to broadcast, a per-edge dict, or a ``(min, max)`` pair to
        remap the absolute edge forces into. If None, uses the default width
        range, so the widths trace the force distribution.

    Returns
    -------
    widths :
        A width for each edge.

    Raises
    ------
    ValueError
        If the width specification is of an unsupported form.
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


def point_colors(
    points: list[int],
    is_support: Callable[[int], bool],
    color: Color | PointColors | None = None,
    default: Color | None = None,
) -> PointColors:
    """
    Map every point to a color, defaulting supports to green.

    Parameters
    ----------
    points :
        The points to color.
    is_support :
        The support predicate of the datastructure.
    color :
        A single color to broadcast or a per-point dict. If None, free points take
        the default and supports turn green.
    default :
        The fallback color of the free points when no ``color`` is given, for
        backends whose canvas needs something other than the grey.

    Returns
    -------
    colors :
        A color for each point.
    """
    if isinstance(color, dict):
        return color

    if isinstance(color, Color):
        return {point: color for point in points}

    default = default if default is not None else COLOR_POINT

    return {point: COLOR_SUPPORT if is_support(point) else default for point in points}


def point_sizes(points: list[int], size: PointSizeSpec = None) -> PointSizes:
    """
    Map every point to a size.

    Parameters
    ----------
    points :
        The points to size.
    size :
        A single size to broadcast or a per-point dict. If None, uses the default
        point size.

    Returns
    -------
    sizes :
        A size for each point.
    """
    if isinstance(size, dict):
        return size

    return {point: size if size is not None else POINT_SIZE for point in points}


def reaction_color_default(edgecolor: EdgeColorSpec) -> Color:
    """
    Pick the default reaction color for a given edge color mode.

    Parameters
    ----------
    edgecolor :
        The edge color mode in effect.

    Returns
    -------
    color :
        Green when edges are colored by force, so reactions read apart from the
        red-blue tension-compression palette; otherwise the default reaction color.
    """
    if edgecolor == "force":
        return Color.from_rgb255(0, 150, 10)
    return COLOR_REACTION


# ==========================================================================
# Arrows
# ==========================================================================


def load_arrows(
    origins: list[list[float]],
    loads: list[list[float]],
    clearances: list[float],
    scale: float,
    tol: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Compute the anchor and vector of the load arrow at every point.

    The arrow is placed so its head touches the point it loads: the anchor
    backs away from the point by the scaled vector, plus the clearance so
    the head clears the edge cylinders.

    Points whose load is below tolerance get a degenerate slot (anchor at the
    point, zero vector) so the arrow count stays constant across updates.

    Parameters
    ----------
    origins :
        The loaded points.
    loads :
        The load vector at every point, aligned with ``origins``.
    clearances :
        The back-off distance at every point (the width of the thickest connected
        edge), aligned with ``origins``.
    scale :
        The factor scaling each load vector to a drawn arrow length.
    tol :
        The magnitude below which a load is treated as zero.

    Returns
    -------
    arrows :
        The anchor points and vectors of the arrows, each aligned with ``origins``.
    """
    anchors, vectors = [], []

    for xyz, vector, clearance in zip(origins, loads, clearances):
        if length_vector(vector) < tol:
            anchors.append(xyz)
            vectors.append((0.0, 0.0, 0.0))
            continue

        # shift start to make the arrow head touch the loaded point,
        # then back off by the clearance so the head clears the edge cylinders
        start = add_vectors(xyz, scale_vector(vector, -scale))
        start = add_vectors(start, scale_vector(normalize_vector(vector), -clearance))

        anchors.append(start)
        vectors.append(scale_vector(vector, scale))

    return anchors, vectors


def reaction_arrows(
    origins: list[list[float]],
    reactions: list[list[float]],
    forces: list[list[float]],
    scale: float,
    tol: float,
    clearances: list[float] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Compute the anchor and vector of the reaction arrow at every point.

    The vector is reversed to display the direction the reaction pushes
    against the structure; when the strongest force among the edges connected
    to a point is compressive, the anchor shifts outward so the arrow points
    at the support from outside.

    Points whose reaction is below tolerance (or without connected edges) get
    a degenerate slot so the arrow count stays constant across updates.

    Parameters
    ----------
    origins :
        The supported points.
    reactions :
        The reaction vector at every point, aligned with ``origins``.
    forces :
        The forces of the edges connected to every point, aligned with ``origins``.
    scale :
        The factor scaling each reaction vector to a drawn arrow length.
    tol :
        The magnitude below which a reaction is treated as zero.
    clearances :
        The gap to keep between the arrow and its point, mirroring the clearance of
        :func:`load_arrows` and aligned with ``origins``: the arrow shifts along its
        drawn direction so its point-touching end (the tail under tension, the tip
        under compression) stops that far short of the point. If None, no gap.

    Returns
    -------
    arrows :
        The anchor points and vectors of the arrows, each aligned with ``origins``.
    """
    if clearances is None:
        clearances = [0.0] * len(origins)

    anchors, vectors = [], []

    for xyz, vector, edge_forces, clearance in zip(
        origins,
        reactions,
        forces,
        clearances,
    ):
        if length_vector(vector) < tol or not edge_forces:
            anchors.append(xyz)
            vectors.append((0.0, 0.0, 0.0))
            continue

        start = xyz
        drawn = scale_vector(vector, -scale)
        if max(edge_forces, key=fabs) < 0.0:
            start = add_vectors(start, scale_vector(vector, scale))
            if clearance:
                start = add_vectors(
                    start,
                    scale_vector(normalize_vector(drawn), -clearance),
                )
        elif clearance:
            start = add_vectors(start, scale_vector(normalize_vector(drawn), clearance))

        anchors.append(start)
        vectors.append(drawn)

    return anchors, vectors
