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
    Calculate the length of a vector along its columns.
    """
    return jnp.linalg.norm(u, axis=-1, keepdims=True)


def normalize_vector(u):
    """
    Scale a vector such that it has a unit length.
    """
    return u / length_vector(u)


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

    A polygon that is defined as a sequence of unique points.
    A polygon must have at least three points.
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

    A polygon that is defined as a sequence of unique points and must be
    defined by at least 3 points.

    This function ignores nan values if any in the input polygon.
    """
    assert len(polygon) >= 3, "A polygon must be defined by at least 3 points"

    polygon_shifted = jnp.roll(polygon, 1, axis=0)
    ns = 0.5 * jnp.cross(polygon_shifted, polygon)
    normal = jnp.nansum(ns, axis=0)

    if unitized:
        return normalize_vector(normal)

    return normal


def area_polygon(polygon):
    """
    Computes the area of a polygon.

    A polygon that is defined as a sequence of unique points and must be
    defined by at least 3 points.

    This function ignores nan values if any in the input polygon.
    """
    return length_vector(normal_polygon(polygon, False))


def normal_triangle(triangle, unitize=False):
    """
    Computes the normal vector of a triangle.

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

    A triangle is defined as a set of exactly three points.
    """
    return 0.5 * length_vector(normal_triangle(triangle))


def _curvature_point_polygon(point, polygon):
    raise NotImplementedError


def _curvature_point_polygon(point, polygon):
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


if __name__ == "__main__":

    from compas.geometry import Polygon
    from compas.geometry import Rotation
    from compas.geometry import Frame
    from compas.geometry import Vector

    from jax import jit

    # Test vector transformation from XYZ to polygon LCS

    # Test polygon LCS
    load = Vector(0.0, 0.0, 1.0)
    load_scale = 2.0
    polygon = Polygon.from_sides_and_radius_xy(4, 1.0)

    polygon_lcs = jit(polygon_lcs)
    lcs = polygon_lcs(jnp.array(polygon.points))
    assert jnp.allclose(lcs, WORLD_XYZ), f"lcs:\n{lcs}\nlcs_target:\n{WORLD_XYZ}"

    load_lcs = jnp.array(load * load_scale) @ lcs
    load_target = WORLD_Z * load_scale
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"
    load_lcs = -jnp.array(load * load_scale) @ lcs
    load_target = load_target * -1.0
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"
    print("Passed world XYZ test\n")

    R = Rotation.from_frame_to_frame(Frame.worldXY(), Frame.worldZX())
    polygon_transformed = polygon.transformed(R)
    lcs = polygon_lcs(jnp.array(polygon_transformed.points[::-1]))
    lcs_target = jnp.vstack((WORLD_X, WORLD_Z, -WORLD_Y))
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"

    load_lcs = jnp.array(load * load_scale) @ lcs
    load_target = -WORLD_Y * load_scale
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"
    load_lcs = -jnp.array(load * load_scale) @ lcs
    load_target = load_target * -1.0
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"

    print("Passed world XY to ZX test\n")

    R = Rotation.from_frame_to_frame(Frame.worldXY(), Frame.worldYZ())
    polygon_transformed = polygon.transformed(R)
    lcs = polygon_lcs(jnp.array(polygon_transformed.points))
    lcs_target = jnp.vstack((WORLD_Y, WORLD_Z, WORLD_X))
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"

    load_lcs = jnp.array(load * load_scale) @ lcs
    load_target = WORLD_X * load_scale
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"
    load_lcs = -jnp.array(load * load_scale) @ lcs
    load_target = load_target * -1.0
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"

    print("Passed world XY to YZ test\n")

    # Test line LCS
    line_lcs = jit(line_lcs)
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    line = jnp.array(line)
    lcs = line_lcs(line)

    lcs_target = WORLD_XYZ
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"

    line = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    line = jnp.array(line)
    lcs = line_lcs(line)

    lcs_target = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"

    line = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    line = jnp.array(line)
    lcs = line_lcs(line)

    lcs_target = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"

    # Test area polygon

    import numpy as np
    from jax import vmap, jit
    from math import pi
    from compas.geometry import Polygon
    from compas.geometry import Rotation
    from compas.geometry import area_polygon as compas_area_polygon
    from compas.geometry import normal_polygon as compas_normal_polygon

    radius = 2.0
    num_angles = 100

    polygons = []

    area_polygon = jit(area_polygon)
    area_polygon(np.ones((5, 3)))

    for i in range(4, 5):
        polygon = Polygon.from_sides_and_radius_xy(i, radius)

        for j in range(1, num_angles):

            polygon_old = polygon

            angle = (j * (0.5 * pi / num_angles))
            R1 = Rotation.from_axis_and_angle(axis=[1.0, 0.0, 0.0], angle=angle)
            polygon = polygon.transformed(R1)
            R2 = Rotation.from_axis_and_angle(axis=[0.0, 1.0, 0.0], angle=angle)
            polygon = polygon.transformed(R2)
            polygons.append(polygon)

            assert not np.allclose(polygon, polygon_old)

            area_compas = compas_area_polygon(polygon.points)
            area = area_polygon(np.array(polygon.points))  # .item()

            assert jnp.allclose(area_compas, area), f"Not equal: JAX: {area:.2f} vs. COMPAS: {area_compas:.2f}"

            normal_compas = compas_normal_polygon(polygon.points)
            normal = normal_polygon(np.array(polygon.points))  # .item()
            assert jnp.allclose(np.array(normal_compas), normal), f"Not equal: JAX: {normal:.2f} vs. COMPAS: {normal_compas:.2f}"

            points_nan = np.reshape(np.array([np.nan] * 6), (-1, 3))
            polygon_nan = np.vstack((points_nan, np.array(polygon.points), points_nan))
            normal_nan = normal_polygon(polygon_nan)

            assert jnp.allclose(np.array(normal_compas), normal_nan), f"Not equal: JAX: {normal_nan:.2f} vs. COMPAS: {normal_compas:.2f}"

    areas = vmap(area_polygon)(jnp.array(polygons))
    normal = vmap(normal_polygon)(jnp.array(polygons))

    print(f"{areas.shape=}")
    print("All good!")
