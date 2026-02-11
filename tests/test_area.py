from math import pi

import numpy as np
import jax.numpy as jnp

from compas.geometry import Polygon
from compas.geometry import Rotation
from compas.geometry import area_polygon as compas_area_polygon

from jax_fdm.geometry import area_polygon


def create_polygons():
    """
    Create polygons with different number of sides.
    """
    polygons = []
    for i in range(3, 10):
        polygon = Polygon.from_sides_and_radius_xy(i, radius=2.0)
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


def compare_areas(points, points_compas):
    """
    Compare the areas of two polygons.
    """
    area_compas = compas_area_polygon(points_compas)
    area = area_polygon(points)
    assert jnp.allclose(area_compas, area)


def test_area_polygon():
    """
    Test the area of polygons with different number of sides.
    """
    for polygon in create_polygons():
        for j in range(1, 4):
            angle = j * (0.5 * pi / 4)
            polygon = rotate_polygon(polygon, angle)
            points = np.array(polygon.points)
            compare_areas(points, points)


def test_area_polygon_padded_start():
    """
    Test the area of polygons padded with the start point.
    """
    for polygon in create_polygons():
        point_start = polygon.points[0]
        polygon_padded = np.vstack((np.array(polygon.points), point_start, point_start))
        compare_areas(polygon_padded, polygon.points)
        polygon_padded = np.vstack((point_start, point_start, np.array(polygon.points)))
        compare_areas(polygon_padded, polygon.points)


def test_area_polygon_padded_end():
    """
    Test the area of polygons padded with the last point.
    """
    for polygon in create_polygons():
        point_end = polygon.points[-1]
        polygon_padded = np.vstack((np.array(polygon.points), point_end, point_end))
        compare_areas(polygon_padded, polygon.points)
        polygon_padded = np.vstack((point_end, np.array(polygon.points), point_end))
        compare_areas(polygon_padded, polygon.points)
