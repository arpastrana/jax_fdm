from jax import vmap

import jax.numpy as jnp

from jax.lax import cond


WORLD_X = jnp.array([1.0, 0.0, 0.0])
WORLD_Y = jnp.array([0.0, 1.0, 0.0])
WORLD_Z = jnp.array([0.0, 0.0, 1.0])
WORLD_XYZ = jnp.eye(3)


def angle_vectors(u, v, deg=False):
    """
    Compute the smallest angle in degrees between two vectors.
    """
    L = length_vector(u) * length_vector(v)
    cosim = jnp.maximum(jnp.minimum((u @ v) / L, 1.0), -1.0)
    angle = jnp.arccos(cosim)

    if deg:
        return jnp.degrees(angle)
    return angle


def length_vector(u):
    """
    Calculate the length of a vector.
    """
    length = jnp.linalg.norm(u, axis=-1, keepdims=True)

    return length


def length_vector_sqrd(u):
    """
    Calculate the squared length of a vector.
    """
    return jnp.sum(u * u, axis=-1, keepdims=True)


def normalize_vector(u, safe_nan=True):
    """
    Scale a vector such that it has a unit length.

    Notes
    -----
    If safe_nan is True, any nan values in the input vector are replaced by zeroes.
    The function will return a zero vector if the input vector is a zero vector
    or if all its elements are nan.
    """
    if safe_nan:
        u = jnp.nan_to_num(u)
        is_zero_vector = jnp.allclose(u, 0.0)
        u = jnp.where(is_zero_vector, jnp.zeros_like(u), u)
        length = jnp.where(is_zero_vector, 1.0, length_vector(u))
    else:
        length = length_vector(u)

    return u / length


def vector_unitized(u):
    """
    Scale a vector such that it has a unit length.
    """
    return u / length_vector(u)


def subtract_vectors(u, v):
    """
    Subtract two vectors.
    """
    return u - v


def line_vector(line, normalized=True):
    """
    Calculate the (normalized) vector formed by the difference of the line end points.
    """
    vector = subtract_vectors(line[1], line[0])
    if normalized:
        return vector_unitized(vector)

    return vector


def vector_projection(u, v):
    """
    Calculates the orthogonal projection of u onto v.
    """
    l2 = jnp.sum(v ** 2)
    x = (u @ jnp.transpose(v)) / l2

    return v * x


def closest_point_on_plane(point, plane):
    """
    Computes the closest location on a plane to a supplied point.
    """
    origin, normal = plane
    normal = normalize_vector(normal)
    d = normal @ origin
    e = normal @ point - d
    k = e / jnp.sum(jnp.square(normal))

    return point - normal * k


def closest_point_on_line(point, line):
    """
    Computes the closest location on a line to a supplied point.
    """
    a, b = line
    ab = b - a
    ap = point - a
    c = vector_projection(ap, ab)

    return a + c


def closest_point_on_segment(point, segment):
    """
    Calculate the closest location on a segment to an input point.
    """
    a, b = segment
    p = closest_point_on_line(point, segment)

    # calculate distances to the three possible points
    d = distance_point_point_sqrd(a, b)
    d1 = distance_point_point_sqrd(a, p)
    d2 = distance_point_point_sqrd(b, p)

    # define callables to be compatible with the signature of lax.cond
    def start_point():
        return a

    def end_point():
        return b

    def middle_point():
        return p

    def point_is_outside():
        return cond(d1 < d2, start_point, end_point)

    # return closest amont the three points
    return cond(jnp.logical_or(d1 > d, d2 > d), point_is_outside, middle_point)


def distance_point_point_sqrd(u, v):
    """
    Calculate the square of the distance between two points.
    """
    vector = subtract_vectors(u, v)

    return jnp.sum(jnp.square(vector))


def normal_polygon_2(polygon, unitized=False):
    """
    Computes the unit-length normal of a polygon.

    Notes
    -----
    A polygon that is defined as a sequence of at least three points.
    """
    centroid = jnp.mean(polygon, axis=0)
    op = polygon - centroid
    op_shifted = jnp.roll(op, 1, axis=0)
    ns = 0.5 * jnp.cross(op_shifted, op)
    n = jnp.sum(ns, axis=0)

    if unitized:
        return normalize_vector(n)
    return n


def normal_polygon(polygon, unitized=True):
    """
    Computes the normal of a closed polygon.

    Notes
    -----
    A polygon that is defined as a sequence of unique points and must be
    defined by at least 3 points. This function is not nan safe, because it
    loses information when cross-multiplying valid entries with nan values.
    """
    assert len(polygon) >= 3, "A polygon must be defined by at least 3 points"

    polygon_shifted = jnp.roll(polygon, 1, axis=0)
    ns = 0.5 * jnp.cross(polygon_shifted, polygon)
    normal = jnp.nansum(ns, axis=0)

    if unitized:
        normal = normalize_vector(normal)

    return normal


def area_polygon(polygon):
    """
    Computes the area of a polygon.

    A polygon that is defined as a sequence of unique points and must be
    defined by at least 3 points.
    """
    return length_vector(normal_polygon(polygon, unitized=False))


def normal_triangle(triangle, unitize=False):
    """
    Computes the normal vector of a triangle.

    Notes
    -----
    A triangle is defined as a set of exactly three points.
    """
    assert len(triangle) == 3, "A triangle is defined by exactly 3 points"

    line_a, line_b = triangle[:-1, :] - triangle[1:, :]
    normal = jnp.cross(line_a, line_b)

    if unitize:
        return normalize_vector(normal)

    return normal


def area_triangle(triangle):
    """
    Calculate the area of a triangle.

    Notes
    -----
    A triangle is defined as a set of exactly three points.
    """
    return 0.5 * length_vector(normal_triangle(triangle))


def planarity_polygon(polygon):
    """
    Calculate the planarity of a polygon.

    Notes
    -----
    The planarity of a polygon is calculated as the sum of the absolute dot product
    between the polygon's unitized normal vector and its unitized edge vectors.
    """
    polygon_shifted = jnp.roll(polygon, 1, axis=0)
    assert polygon.shape == polygon_shifted.shape

    edge_vectors = polygon - polygon_shifted
    unit_vectors = vmap(normalize_vector, in_axes=(0,))(edge_vectors)
    assert unit_vectors.shape == edge_vectors.shape

    normal = normal_polygon(polygon)
    cosines = vmap(jnp.dot, in_axes=(0, None))(unit_vectors, normal)
    assert cosines.shape[0] == polygon.shape[0]

    planarity = jnp.sum(jnp.abs(cosines))

    return planarity


def planarity_triangle(triangle):
    """
    Calculate the planarity of a triangle.

    Notes
    -----
    The planarity of a triangle is 0.0 by construction.
    """
    return 0.0


def curvature_point_polygon(point, polygon):
    """
    Compute the discrete curvature at a point based on a polygon surrounding it.
    The discrete curvature of a node equals 2 * pi - sum(alphas).

    TODO: divide return value by tributary area of point.

    Notes
    -----
    Alphas is the list of angles between each pair of successive edges as
    the outward vectors from the node. Polygon is a numpy array (# points, 3).
    """
    op = polygon - point
    norm_op = jnp.reshape(jnp.linalg.norm(op, axis=1), (-1, 1))
    op = op / norm_op
    op_off = jnp.roll(op, 1, axis=0)
    dot = jnp.sum(op_off * op, axis=1)
    dot = jnp.maximum(jnp.minimum(dot, jnp.full(dot.shape, 1)), jnp.full(dot.shape, -1))
    angles = jnp.arccos(dot)

    return 2 * jnp.pi - jnp.sum(angles)


def line_lcs(line):
    """
    Returns the local coordinate system (LCS) of a line.

    Notes
    -----
    The LCS is a 3D orthonormal basis formed by unit vectors U, V, and W.
    The orthonormal basis is constructed using the following convention:

    - The U axis is the unit vector of the coordinate difference of the line end points.
    - The V axis is the cross product of U and the global Z axis. If the V axis is
      parallel to Z, then V becomes the cross product of U and the global Y axis.
    - The W axis is the cross product of U and V.
    """
    u = line_vector(line)

    threshold = jnp.allclose(jnp.abs(WORLD_Z @ u), 1.0)
    vperp = jnp.where(threshold, WORLD_Y, WORLD_Z)
    v = jnp.cross(vperp, u)

    w = jnp.cross(u, v)
    w = jnp.where(w @ vperp < 0.0, -w, w)

    return jnp.vstack((u, v, w))


def polygon_lcs(polygon):
    """
    Returns the local coordinate system (LCS) of a polygon.

    Notes
    -----
    The LCS is a 3D orthonormal basis formed by unit vectors U, V, and W.
    The orthonormal basis is constructed using the following convention:

    - The W axis is the polygon normal.
    - The U axis is the unit vector of the coordinate difference of the line end points.
    - The V axis is the cross product of U and the global Z axis. If the V axis is
        parallel to Z, then V becomes the cross product of U and the global Y axis.
    """
    w = normal_polygon(polygon, True)

    threshold = jnp.allclose(jnp.abs(WORLD_X @ w), 1.0)
    vperp = jnp.where(threshold, WORLD_Y, WORLD_X)
    v = jnp.cross(w, vperp)

    u = jnp.cross(w, v)
    u = jnp.where(u @ vperp < 0.0, -u, u)

    return jnp.vstack((u, v, w))
