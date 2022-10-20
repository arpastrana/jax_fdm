import jax.numpy as jnp

from jax.lax import cond


def angle_vectors(u, v):
    """
    Compute the smallest angle in degrees between two vectors.
    """
    L = length_vector(u) * length_vector(v)
    cosim = (u @ v) / L
    cosim = jnp.maximum(jnp.minimum(cosim, 1.0), -1.0)

    return jnp.degrees(jnp.arccos(cosim))


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


def subtract_vectors(u, v):
    """
    Subtract two vectors.
    """
    return u - v


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


def normal_polygon(polygon):
    """
    Computes the unit-length normal of a polygon defined by a sequence of points.

    A polygon must have at least two points.
    """
    centroid = jnp.mean(polygon, axis=0)
    op = polygon - centroid
    op_off = jnp.roll(op, 1, axis=0)
    ns = jnp.multiply(jnp.cross(op_off, op), 0.5)
    n = jnp.sum(ns, axis=0)

    return normalize_vector(n)


def curvature_point_polygon(point, polygon):
    """
    Compute the discrete curvature at a point based on a polygon surrounding it.
    The discrete curvature of a node equals 2 * pi - sum(alphas).

    Notes
    -----
    Alphas is the list of angles between each pair of successive edges as outward vectors from the node.
    Polygon is numpy array #points x 3.
    """
    op = polygon - point
    norm_op = jnp.reshape(jnp.linalg.norm(op, axis=1), (-1, 1))
    op = op / norm_op
    op_off = jnp.roll(op, 1, axis=0)
    dot = jnp.sum(op_off * op, axis=1)
    dot = jnp.maximum(jnp.minimum(dot, jnp.full(dot.shape, 1)), jnp.full(dot.shape, -1))
    angles = jnp.arccos(dot)

    return 2 * jnp.pi - jnp.sum(angles)


if __name__ == "__main__":
    from jax import jit

    point_a = [1.0, 0.0, 0.0]
    point_b = [2.5, 0.0, 0.0]

    segment = tuple([jnp.array(point) for point in (point_a, point_b)])

    test_points = [[2.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [5.0, -1.0, 0.0]]

    result_points = [[2.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [2.5, 0.0, 0.0]
                     ]

    for tpoint, rpoint in zip(test_points, result_points):
        cpoint = jit(closest_point_on_segment)(jnp.array(tpoint), segment)
        assert jnp.allclose(cpoint, jnp.array(rpoint)), f"got {cpoint} wanted {rpoint}"

    print("okay!")
