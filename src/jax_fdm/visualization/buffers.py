from functools import lru_cache

import numpy as np
from jaxtyping import Float
from jaxtyping import Int
from numpy.typing import ArrayLike

from compas.geometry import Cylinder
from compas.geometry import Sphere
from jax_fdm.visualization.shapes import Arrow

Soup = tuple[Float[np.ndarray, "vertices 3"], Float[np.ndarray, "vertices 4"]]

# A triangle soup (positions, rgba colors) paired with its flat vertex indices,
# as read out for a viewer buffer.
FacesData = tuple[
    Float[np.ndarray, "vertices 3"],
    Float[np.ndarray, "vertices 4"],
    Int[np.ndarray, "vertices"],
]

__all__ = [
    "cylinders_buffer",
    "arrows_buffer",
    "spheres_buffer",
    "soup_indices",
    "soup_colors_rgb",
]


# ==========================================================================
# Unit templates
# ==========================================================================

# The templates are canonical shapes tessellated once and expanded to a
# triangle soup (one row of xyz per triangle corner), so that a whole
# collection of elements becomes template * scale @ rotation + translation,
# fully vectorized. Templates are cached per resolution.


@lru_cache(maxsize=None)
def cylinder_template(u: int = 16) -> Float[np.ndarray, "vertices 3"]:
    """
    The triangle soup of a unit cylinder (radius 1, height 1, centered at
    the origin, axis along z), as an array of shape (T * 3, 3).
    """
    vertices, faces = Cylinder(radius=1.0, height=1.0).to_vertices_and_faces(
        u=u,
        triangulated=True,
    )
    return np.asarray(vertices, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64).ravel()
    ]


@lru_cache(maxsize=None)
def sphere_template(u: int = 16, v: int = 16) -> Float[np.ndarray, "vertices 3"]:
    """
    The triangle soup of a unit sphere centered at the origin,
    as an array of shape (T * 3, 3).
    """
    vertices, faces = Sphere(radius=1.0).to_vertices_and_faces(
        u=u,
        v=v,
        triangulated=True,
    )
    return np.asarray(vertices, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64).ravel()
    ]


@lru_cache(maxsize=None)
def arrow_template(
    u: int = 8,
    head_portion: float = 0.12,
    head_width: float = 0.04,
    body_width: float = 0.012,
) -> Float[np.ndarray, "vertices 3"]:
    """
    The triangle soup of a unit arrow anchored at the origin and pointing
    along z, as an array of shape (T * 3, 3).

    An arrow scales uniformly with its length (the head and body widths are
    fractions of the length), so any arrow is this template under a uniform
    scale, a rotation and a translation.
    """
    arrow = Arrow(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        head_portion=head_portion,
        head_width=head_width,
        body_width=body_width,
    )
    vertices, faces = arrow.to_vertices_and_faces(u=u, triangulated=True)
    return np.asarray(vertices, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64).ravel()
    ]


# ==========================================================================
# Vectorized transforms
# ==========================================================================


def rotations_to(
    directions: Float[np.ndarray, "elements 3"],
) -> Float[np.ndarray, "elements 3 3"]:
    """
    Per-row rotation matrices mapping the z axis onto each direction.

    The columns of every matrix are the frame axes, so R @ [0, 0, 1] equals
    the (unit) direction. The reference axis for the cross product is the
    cardinal axis least aligned with each direction, which keeps the cross
    product well-conditioned and handles directions parallel to z.

    Parameters
    ----------
    directions :
        Unit direction vectors.

    Returns
    -------
    rotations :
        One rotation matrix per direction.
    """
    z = np.asarray(directions, dtype=np.float64)
    ref = np.zeros_like(z)
    ref[np.arange(len(z)), np.argmin(np.abs(z), axis=1)] = 1.0

    x = np.cross(ref, z)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)

    return np.stack((x, y, z), axis=-1)


def _soup(
    template: Float[np.ndarray, "vertices 3"],
    rotations: Float[np.ndarray, "elements 3 3"],
    scales: Float[np.ndarray, "elements 3"],
    translations: Float[np.ndarray, "elements 3"],
    colors: ArrayLike,
) -> Soup:
    """
    Transform a template per element and expand colors per soup vertex.

    Returns positions of shape (N * T * 3, 3) as float32 and colors of shape
    (N * T * 3, 4) as float32, the layout the viewer buffer objects expect.
    """
    local = template[None, :, :] * scales[:, None, :]
    world = np.einsum("nij,ntj->nti", rotations, local) + translations[:, None, :]

    positions = world.reshape(-1, 3).astype(np.float32)
    facecolors = np.repeat(
        np.asarray(colors, dtype=np.float32),
        template.shape[0],
        axis=0,
    )

    return positions, facecolors


def _empty_buffer() -> Soup:
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)


# ==========================================================================
# Soup views
# ==========================================================================


def soup_indices(soup: Soup, flipped: bool = False) -> Int[np.ndarray, "vertices"]:
    """
    The vertex indices of a soup, optionally with flipped winding.

    Parameters
    ----------
    soup : (positions, colors)
        A triangle soup, as built by the buffer builders.
    flipped : bool, optional
        Reverse the indices, flipping the winding of every triangle.

    Returns
    -------
    array of shape (N * T * 3,)
    """
    positions, _ = soup
    indices = np.arange(len(positions))
    return np.flip(indices) if flipped else indices


def soup_colors_rgb(soup: Soup) -> Float[np.ndarray, "vertices 3"]:
    """
    The colors of a soup stripped to rgb, as a contiguous array.

    The buffer builders emit rgba; some consumers (e.g. pythreejs buffer
    attributes) take rgb.

    Parameters
    ----------
    soup : (positions, colors)
        A triangle soup, as built by the buffer builders.

    Returns
    -------
    array of shape (N * T * 3, 3), float32
    """
    _, colors = soup
    return np.ascontiguousarray(colors[:, :3])


# ==========================================================================
# Buffer builders
# ==========================================================================


def cylinders_buffer(
    starts: ArrayLike,
    ends: ArrayLike,
    radii: ArrayLike,
    colors: ArrayLike,
    u: int = 16,
) -> Soup:
    """
    Batch cylinders spanning pairs of points into one triangle soup.

    Parameters
    ----------
    starts, ends : array of shape (N, 3)
        The endpoints of every cylinder axis.
    radii : array of shape (N,)
    colors : array of shape (N, 4)
        One rgba color per cylinder.
    u : int, optional
        Resolution of the template tessellation.

    Returns
    -------
    (positions, colors)
        Arrays of shape (N * T * 3, 3) and (N * T * 3, 4), float32.

    Notes
    -----
    Zero-length cylinders collapse to a degenerate soup at their start point,
    keeping the vertex count constant so the render buffers update in place.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    if len(starts) == 0:
        return _empty_buffer()

    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)

    live = lengths > 0.0
    directions = np.tile([0.0, 0.0, 1.0], (len(starts), 1))
    directions[live] = vectors[live] / lengths[live, None]

    scales = np.stack((radii, radii, lengths), axis=1)
    scales[~live] = 0.0

    rotations = rotations_to(directions)
    midpoints = (starts + ends) / 2.0

    return _soup(cylinder_template(u), rotations, scales, midpoints, colors)


def arrows_buffer(
    anchors: ArrayLike,
    vectors: ArrayLike,
    colors: ArrayLike,
    head_portion: float = 0.12,
    head_width: float = 0.04,
    body_width: float = 0.012,
    u: int = 8,
    tol: float = 1e-12,
) -> Soup:
    """
    Batch arrows anchored at points into one triangle soup.

    Parameters
    ----------
    anchors : array of shape (N, 3)
        The points every arrow starts from.
    vectors : array of shape (N, 3)
        The arrow vectors. Vectors shorter than ``tol`` yield a degenerate
        arrow collapsed at its anchor.
    colors : array of shape (N, 4)
        One rgba color per arrow.
    head_portion, head_width, body_width : float, optional
        Arrow proportions, as fractions of the arrow length.
    u : int, optional
        Resolution of the template tessellation.

    Returns
    -------
    (positions, colors)
        Arrays of shape (N * T * 3, 3) and (N * T * 3, 4), float32.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)

    if len(anchors) == 0:
        return _empty_buffer()

    lengths = np.linalg.norm(vectors, axis=1)

    live = lengths > tol
    directions = np.tile([0.0, 0.0, 1.0], (len(anchors), 1))
    directions[live] = vectors[live] / lengths[live, None]

    # arrows scale uniformly: head and body widths are fractions of length
    scales = np.repeat(lengths[:, None], 3, axis=1)
    scales[~live] = 0.0

    rotations = rotations_to(directions)
    template = arrow_template(u, head_portion, head_width, body_width)

    return _soup(template, rotations, scales, anchors, colors)


def spheres_buffer(
    centers: ArrayLike,
    radii: ArrayLike,
    colors: ArrayLike,
    u: int = 16,
    v: int = 16,
) -> Soup:
    """
    Batch spheres centered at points into one triangle soup.

    Parameters
    ----------
    centers : array of shape (N, 3)
    radii : array of shape (N,)
    colors : array of shape (N, 4)
        One rgba color per sphere.
    u, v : int, optional
        Resolution of the template tessellation.

    Returns
    -------
    (positions, colors)
        Arrays of shape (N * T * 3, 3) and (N * T * 3, 4), float32.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    if len(centers) == 0:
        return _empty_buffer()

    scales = np.repeat(radii[:, None], 3, axis=1)
    rotations = np.tile(np.eye(3), (len(centers), 1, 1))

    return _soup(sphere_template(u, v), rotations, scales, centers, colors)
