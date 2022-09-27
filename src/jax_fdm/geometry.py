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
    Computes the normal of a polygon defined by a sequence of points.

    A polygon must have at least two points.
    """
    centroid = jnp.mean(polygon, axis=0)
    op = polygon - centroid
    # TODO: vectorize ns, may cause jit unecessary loop-unrolling
    ns = jnp.array([jnp.cross(op[i - 1], op[i]) * 0.5 for i in range(len(op))])
    n = jnp.sum(ns, axis=0)
    return normalize_vector(n)
