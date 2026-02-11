from math import pi
from math import sqrt

import numpy as np
import jax.numpy as jnp

from compas.geometry import Polygon
from compas.geometry import Rotation
from compas.geometry import normal_polygon as compas_normal_polygon

from jax_fdm.geometry import normal_polygon


def create_polygons():
    """
    Create polygons with different number of sides.
    """
    polygons = []
    for i in range(4, 10):
        polygon = Polygon.from_sides_and_radius_xy(i, radius=1.0)
        polygons.append(polygon)
    return polygons


def rotate_polygon(polygon, angle):
    """
    Rotate a polygon by a given angle around the x and y axes.
    """
    R1 = Rotation.from_axis_and_angle(axis=[1.0, 0.0, 0.0], angle=angle)
    polygon = polygon.transformed(R1)
    R2 = Rotation.from_axis_and_angle(axis=[0.0, 1.0, 0.0], angle=angle)
    polygon = polygon.transformed(R2)

    return polygon


def compare_normals(points, points_compas, unitized=True):
    """
    Compare the normal vectors of two polygons.
    """
    normal_compas = compas_normal_polygon(points_compas, unitized=unitized)
    normal = normal_polygon(points, unitized=unitized)

    assert jnp.allclose(jnp.array(normal_compas), jnp.array(normal))


def test_normal_polygon():
    """
    Test the normal vector of polygons with different number of sides.
    """
    num_angles = 4
    for polygon in create_polygons():
        for j in range(1, num_angles):
            angle = j * (0.5 * pi / num_angles)
            polygon = rotate_polygon(polygon, angle)
            points = np.array(polygon.points)
            compare_normals(points, points)
            compare_normals(points, points, unitized=False)


def test_normal_polygon_padded_start():
    """
    Test the normal vector of polygons padded with the start point.
    """
    for polygon in create_polygons():
        point_start = polygon.points[0]
        polygon_padded = np.vstack((np.array(polygon.points), point_start, point_start))
        compare_normals(polygon_padded, polygon.points)
        compare_normals(polygon_padded, polygon.points, unitized=False)

        polygon_padded = np.vstack((point_start, point_start, np.array(polygon.points)))
        compare_normals(polygon_padded, polygon.points)
        compare_normals(polygon_padded, polygon.points, unitized=False)


def test_normal_polygon_padded_end():
    """
    Test the normal vector of polygons padded with the end point.
    """
    for polygon in create_polygons():
        point_end = polygon.points[-1]
        polygon_padded = np.vstack((np.array(polygon.points), point_end, point_end))
        compare_normals(polygon_padded, polygon.points)
        compare_normals(polygon_padded, polygon.points, unitized=False)

        polygon_padded = np.vstack((point_end, np.array(polygon.points), point_end))
        compare_normals(polygon_padded, polygon.points)
        compare_normals(polygon_padded, polygon.points, unitized=False)


def test_normal_polygon_nan():
    """
    Test the normal vector of polygons with nan values.
    """
    for polygon in create_polygons():
        # Inject 2 rows of nan values at the end of the polygon
        points_nan = np.reshape(np.array([np.nan] * 6), (2, 3))
        polygon_nan = np.vstack((np.array(polygon.points), points_nan))

        # Test unitized normal vector
        compare_normals(polygon_nan, polygon.points)

        # Test non-unitized normal vector
        normal_compas = compas_normal_polygon(polygon.points, unitized=False)
        normal_nan = normal_polygon(polygon_nan, unitized=False)
        assert not jnp.allclose(np.array(normal_compas), normal_nan)
