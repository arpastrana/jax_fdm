import jax.numpy as jnp

from jax.lax import cond


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


def normal_polygon(polygon, unitize=False):
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

    if unitize:
        return normalize_vector(n)
    return n


def area_polygon(polygon):
    """
    Computes the area of a polygon.

    A polygon that is defined as a sequence of unique points.
    A polygon must have at least three points.
    """
    # breakpoint()
    polygon_shifted = jnp.roll(polygon, 1, axis=0)
    ns = 0.5 * jnp.cross(polygon_shifted, polygon)
    normal = jnp.sum(ns, axis=0)

    return length_vector(normal)


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

    import numpy as np
    from jax import vmap, jit
    from math import pi
    from compas.geometry import Polygon
    from compas.geometry import Rotation
    from compas.geometry import area_polygon as compas_area_polygon

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

            assert jnp.allclose(area_compas, area), f"Not equal: Jax fdm: {area:.2f} vs. Compas {area_compas:.2f}"

    areas = vmap(area_polygon)(jnp.array(polygons))
    print(areas.shape)
    print("Okay")
