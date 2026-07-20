import jax.numpy as jnp
from jax import vmap
from jax.lax import cond
from jaxtyping import Array
from jaxtyping import Float

WORLD_X = jnp.array([1.0, 0.0, 0.0])
WORLD_Y = jnp.array([0.0, 1.0, 0.0])
WORLD_Z = jnp.array([0.0, 0.0, 1.0])
WORLD_XYZ = jnp.eye(3)


def cosine_vectors(u: Float[Array, "3"], v: Float[Array, "3"]) -> Float[Array, ""]:
    """
    Compute the signed cosine of the angle between two vectors.

    Parameters
    ----------
    u :
        The first vector.
    v :
        The second vector.

    Returns
    -------
    cosine :
        The cosine of the angle, in the range [-1, 1].
    """
    return normalize_vector(u) @ normalize_vector(v)


def angle_vectors(
    u: Float[Array, "3"],
    v: Float[Array, "3"],
    deg: bool = False,
) -> Float[Array, ""]:
    """
    Compute the smallest angle between two vectors.

    Parameters
    ----------
    u :
        The first vector.
    v :
        The second vector.
    deg :
        If True, return the angle in degrees instead of radians.

    Returns
    -------
    angle :
        The smallest angle between the vectors.

    Notes
    -----
    The cosine is clipped to [-1, 1] before the arccosine, whose value and
    gradient are singular when the vectors are parallel: in floating point
    the cosine of two parallel vectors can overshoot 1 by a few ulps.
    """
    cosim = cosine_vectors(u, v)
    angle = jnp.arccos(jnp.clip(cosim, -1.0, 1.0))

    if deg:
        return jnp.degrees(angle)

    return angle


def length_vector(u: Float[Array, "3"]) -> Float[Array, "1"]:
    """
    Calculate the length of a vector.

    Parameters
    ----------
    u :
        The vector.

    Returns
    -------
    length :
        The Euclidean length of the vector.
    """
    length = jnp.linalg.norm(u, axis=-1, keepdims=True)

    return length


def length_vector_sqrd(u: Float[Array, "3"]) -> Float[Array, "1"]:
    """
    Calculate the squared length of a vector.

    Parameters
    ----------
    u :
        The vector.

    Returns
    -------
    length :
        The squared Euclidean length of the vector.
    """
    return jnp.sum(u * u, axis=-1, keepdims=True)


def normalize_vector(u: Float[Array, "3"], safe_nan: bool = True) -> Float[Array, "3"]:
    """
    Scale a vector such that it has a unit length.

    Parameters
    ----------
    u :
        The vector to normalize.
    safe_nan :
        If True, replace nan entries by zeroes before normalizing.

    Returns
    -------
    vector :
        The unit-length vector.

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


def vector_unitized(u: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Scale a vector such that it has a unit length.

    Parameters
    ----------
    u :
        The vector to normalize.

    Returns
    -------
    vector :
        The unit-length vector.

    Notes
    -----
    Unlike normalize_vector, this helper does not guard against zero-length or
    nan input, so it divides by the raw vector length.
    """
    return u / length_vector(u)


def subtract_vectors(u: Float[Array, "3"], v: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Subtract two vectors.

    Parameters
    ----------
    u :
        The vector to subtract from.
    v :
        The vector to subtract.

    Returns
    -------
    vector :
        The difference u - v.
    """
    return u - v


def line_vector(
    line: Float[Array, "2 3"],
    normalized: bool = True,
) -> Float[Array, "3"]:
    """
    Calculate the (normalized) vector formed by the difference of the line end points.

    Parameters
    ----------
    line :
        The line, as its two end points.
    normalized :
        If True, scale the vector to unit length.

    Returns
    -------
    vector :
        The vector from the first end point to the second.
    """
    vector = subtract_vectors(line[1], line[0])
    if normalized:
        return vector_unitized(vector)

    return vector


def vector_projection(u: Float[Array, "3"], v: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Calculates the orthogonal projection of u onto v.

    Parameters
    ----------
    u :
        The vector to project.
    v :
        The vector to project onto.

    Returns
    -------
    projection :
        The component of u parallel to v.
    """
    l2 = jnp.sum(v**2)
    x = (u @ jnp.transpose(v)) / l2

    return v * x


def closest_point_on_plane(
    point: Float[Array, "3"],
    plane: Float[Array, "2 3"],
) -> Float[Array, "3"]:
    """
    Computes the closest location on a plane to a supplied point.

    Parameters
    ----------
    point :
        The query point.
    plane :
        The plane, as its origin and normal.

    Returns
    -------
    point :
        The orthogonal projection of the query point onto the plane.
    """
    origin, normal = plane
    normal = normalize_vector(normal)
    d = normal @ origin
    e = normal @ point - d
    k = e / jnp.sum(jnp.square(normal))

    return point - normal * k


def closest_point_on_line(
    point: Float[Array, "3"],
    line: Float[Array, "2 3"],
) -> Float[Array, "3"]:
    """
    Computes the closest location on a line to a supplied point.

    Parameters
    ----------
    point :
        The query point.
    line :
        The line, as its two end points.

    Returns
    -------
    point :
        The orthogonal projection of the query point onto the infinite line.
    """
    a, b = line
    ab = b - a
    ap = point - a
    c = vector_projection(ap, ab)

    return a + c


def closest_point_on_segment(
    point: Float[Array, "3"],
    segment: Float[Array, "2 3"],
) -> Float[Array, "3"]:
    """
    Calculate the closest location on a segment to an input point.

    Parameters
    ----------
    point :
        The query point.
    segment :
        The segment, as its two end points.

    Returns
    -------
    point :
        The closest location on the segment, clamped to its end points.
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


def distance_point_point_sqrd(
    u: Float[Array, "3"],
    v: Float[Array, "3"],
) -> Float[Array, ""]:
    """
    Calculate the square of the distance between two points.

    Parameters
    ----------
    u :
        The first point.
    v :
        The second point.

    Returns
    -------
    distance :
        The squared Euclidean distance between the points.
    """
    vector = subtract_vectors(u, v)

    return jnp.sum(jnp.square(vector))


def normal_polygon(
    polygon: Float[Array, "points 3"],
    unitized: bool = True,
) -> Float[Array, "3"]:
    """
    Computes the normal of a closed polygon.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three unique points.
    unitized :
        If True, scale the normal to unit length.

    Returns
    -------
    normal :
        The polygon normal.

    Notes
    -----
    The polygon points are referenced to their centroid before the cross
    products. For a closed loop the reference point telescopes out of the
    summed cross products, so the normal is unchanged analytically, but the
    centering avoids the catastrophic cancellation that crossing large
    absolute coordinates incurs far away from the origin.

    Rows that contain nan values are excluded from the centroid and drop the
    two cross-product terms they touch, so the direction of the normal
    survives nan padding but its magnitude does not.
    """
    assert len(polygon) >= 3, "A polygon must be defined by at least 3 points"

    centroid = jnp.nanmean(polygon, axis=0)
    polygon = polygon - centroid
    polygon_shifted = jnp.roll(polygon, 1, axis=0)
    ns = 0.5 * jnp.cross(polygon_shifted, polygon)
    normal = jnp.nansum(ns, axis=0)

    if unitized:
        normal = normalize_vector(normal)

    return normal


def area_polygon(polygon: Float[Array, "points 3"]) -> Float[Array, ""]:
    """
    Computes the area of a polygon.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three unique points.

    Returns
    -------
    area :
        The area of the polygon.
    """
    return jnp.squeeze(length_vector(normal_polygon(polygon, unitized=False)))


def normal_triangle(
    triangle: Float[Array, "3 3"],
    unitize: bool = False,
) -> Float[Array, "3"]:
    """
    Computes the normal vector of a triangle.

    Parameters
    ----------
    triangle :
        The triangle, as exactly three points.
    unitize :
        If True, scale the normal to unit length.

    Returns
    -------
    normal :
        The triangle normal.
    """
    assert len(triangle) == 3, "A triangle is defined by exactly 3 points"

    line_a, line_b = triangle[:-1, :] - triangle[1:, :]
    normal = jnp.cross(line_a, line_b)

    if unitize:
        return normalize_vector(normal)

    return normal


def area_triangle(triangle: Float[Array, "3 3"]) -> Float[Array, ""]:
    """
    Calculate the area of a triangle.

    Parameters
    ----------
    triangle :
        The triangle, as exactly three points.

    Returns
    -------
    area :
        The area of the triangle.
    """
    return 0.5 * jnp.squeeze(length_vector(normal_triangle(triangle)))


def planarity_polygon(polygon: Float[Array, "points 3"]) -> Float[Array, ""]:
    """
    Calculate the planarity of a polygon.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three points.

    Returns
    -------
    planarity :
        The planarity energy, zero when the polygon is planar.

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


def planarity_triangle(triangle: Float[Array, "3 3"]) -> Float[Array, ""]:
    """
    Calculate the planarity of a triangle.

    Parameters
    ----------
    triangle :
        The triangle, as exactly three points.

    Returns
    -------
    planarity :
        The planarity energy, always zero for a triangle.
    """
    return jnp.asarray(0.0)


def curvature_point_polygon(
    point: Float[Array, "3"],
    polygon: Float[Array, "points 3"],
) -> Float[Array, ""]:
    """
    Compute the discrete curvature at a point based on a polygon surrounding it.

    The discrete curvature of a node equals 2 * pi - sum(alphas).

    TODO: divide return value by tributary area of point.

    Parameters
    ----------
    point :
        The central point at which the curvature is evaluated.
    polygon :
        The ring of points surrounding the central point.

    Returns
    -------
    curvature :
        The discrete curvature at the central point.

    Notes
    -----
    Alphas is the list of angles between each pair of successive edges as
    the outward vectors from the node.
    """
    op = polygon - point
    norm_op = jnp.reshape(jnp.linalg.norm(op, axis=1), (-1, 1))
    op = op / norm_op
    op_off = jnp.roll(op, 1, axis=0)
    dot = jnp.sum(op_off * op, axis=1)
    dot = jnp.maximum(jnp.minimum(dot, jnp.full(dot.shape, 1)), jnp.full(dot.shape, -1))
    angles = jnp.arccos(dot)

    return 2 * jnp.pi - jnp.sum(angles)


def line_lcs(line: Float[Array, "2 3"]) -> Float[Array, "3 3"]:
    """
    Returns the local coordinate system (LCS) of a line.

    Parameters
    ----------
    line :
        The line, as its two end points.

    Returns
    -------
    lcs :
        The orthonormal basis, as its stacked U, V, and W axes.

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
    v = vector_unitized(v)

    w = jnp.cross(u, v)
    w = jnp.where(w @ vperp < 0.0, -w, w)
    w = vector_unitized(w)

    return jnp.vstack((u, v, w))


def polygon_lcs(polygon: Float[Array, "points 3"]) -> Float[Array, "3 3"]:
    """
    Returns the local coordinate system (LCS) of a polygon.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three points.

    Returns
    -------
    lcs :
        The orthonormal basis, as its stacked U, V, and W axes.

    Notes
    -----
    The LCS is a 3D orthonormal basis formed by unit vectors U, V, and W.
    The orthonormal basis is constructed using the following convention:

    - The W axis is the unitized polygon normal.
    - The V axis is the cross product of W and the global X axis. If W is
      parallel to X, then V becomes the cross product of W and the global Y axis.
    - The U axis is the cross product of W and V, sign-aligned so it points
      away from the reference axis used for V.
    """
    w = normal_polygon(polygon, True)

    threshold = jnp.allclose(jnp.abs(WORLD_X @ w), 1.0)
    vperp = jnp.where(threshold, WORLD_Y, WORLD_X)
    v = jnp.cross(w, vperp)
    v = vector_unitized(v)

    u = jnp.cross(w, v)
    u = jnp.where(u @ vperp < 0.0, -u, u)
    u = vector_unitized(u)

    return jnp.vstack((u, v, w))


def colinearity_points(points: Float[Array, "points 3"]) -> Float[Array, ""]:
    """
    Calculate the colinearity of a sequence of points.

    Parameters
    ----------
    points :
        The ordered sequence of points.

    Returns
    -------
    colinearity :
        The colinearity energy, zero when the points are colinear.

    Notes
    -----
    Colinearity is defined as length-normalized fairness energy:
        E = sum_i ||e_i - e_{i-1}||^2 / (0.5 * (||e_{i-1}||^2 + ||e_i||^2))
    where e_i = points[i+1] - points[i].

    The normalization makes this energy less sensitive to local point spacing
    than a raw second-difference energy. A colinearity of 0.0 indicates that
    the points are colinear (i.e., they lie on a straight line). The result
    is normalized by the number of interior vertices so it is invariant to
    problem size.
    """
    line_vectors = subtract_vectors(points[1:], points[:-1])
    lengths_squared = length_vector_sqrd(line_vectors)

    dt = line_vectors[1:] - line_vectors[:-1]
    dtdt = jnp.sum(dt**2, axis=-1)
    lbar = 0.5 * (lengths_squared[1:] + lengths_squared[:-1])
    n_interior = dt.shape[0]

    return jnp.sum(dtdt / lbar) / jnp.maximum(n_interior, 1)


def curvature_points(points: Float[Array, "points 3"]) -> Float[Array, ""]:
    """
    Compute the curvature (turning) energy of a sequence of points.

    Penalizes changes in direction between consecutive edges. Scale-invariant.

    Parameters
    ----------
    points :
        The ordered sequence of points.

    Returns
    -------
    energy :
        The turning energy, zero when the points are colinear.

    Notes
    -----
    Curvature is defined as raw turning energy:
        E = sum_i ||t_i - t_{i-1}||^2 / (num_points - 2)
    where t_i is the unit tangent vector. Energy depends only on turn angles,
    not on edge lengths, so it is scale-invariant.
    """
    line_vectors = subtract_vectors(points[1:], points[:-1])
    tangents = vector_unitized(line_vectors)

    dt = tangents[1:] - tangents[:-1]
    n_interior = dt.shape[0]

    return jnp.sum(dt**2) / jnp.maximum(n_interior, 1)


def _unit_edge_vectors_polygon(
    polygon: Float[Array, "points 3"],
) -> tuple[Float[Array, "points 3"], Float[Array, "points 3"]]:
    """
    Return unit vectors from each vertex to its prev and next neighbors.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three points.

    Returns
    -------
    unit_to_prev, unit_to_next :
        The unit vectors from each vertex toward its previous and next
        neighbor, one pair per vertex.
    """
    polygon_prev = jnp.roll(polygon, shift=1, axis=0)
    polygon_next = jnp.roll(polygon, shift=-1, axis=0)

    edge_to_prev = polygon_prev - polygon  # from i toward i-1
    edge_to_next = polygon_next - polygon  # from i toward i+1

    unit_to_prev = vmap(normalize_vector, in_axes=(0,))(edge_to_prev)
    unit_to_next = vmap(normalize_vector, in_axes=(0,))(edge_to_next)

    return unit_to_prev, unit_to_next


def angles_polygon(
    polygon: Float[Array, "points 3"],
    deg: bool = False,
) -> Float[Array, "points"]:
    """
    Calculate the internal angles of a polygon.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three points.
    deg :
        If True, return the angles in degrees instead of radians.

    Returns
    -------
    angles :
        The internal angle at each polygon vertex.
    """
    unit_to_prev, unit_to_next = _unit_edge_vectors_polygon(polygon)

    return vmap(angle_vectors, in_axes=(0, 0, None))(unit_to_prev, unit_to_next, deg)


def cosines_angles_polygon(polygon: Float[Array, "points 3"]) -> Float[Array, "points"]:
    """
    Calculate the internal angle cosines of a polygon.

    Parameters
    ----------
    polygon :
        The polygon, as a sequence of at least three points.

    Returns
    -------
    cosines :
        The cosine of the internal angle at each polygon vertex.
    """
    unit_to_prev, unit_to_next = _unit_edge_vectors_polygon(polygon)

    cosines = vmap(cosine_vectors, in_axes=(0, 0))(unit_to_prev, unit_to_next)

    return cosines
