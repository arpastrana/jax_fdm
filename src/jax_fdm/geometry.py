import jax.numpy as jnp


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
