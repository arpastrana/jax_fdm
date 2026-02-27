"""
Unit tests for angles_polygon.
"""
from itertools import cycle
import jax
import jax.numpy as jnp
from jax import vmap

from compas.geometry import Polygon
from compas.geometry import Rotation

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.geometry import angles_polygon
from jax_fdm.geometry import cosines_angles_polygon


# Analytical internal angles for regular n-gons: (n-2)*π/n radians
TRIANGLE_ANGLE_RAD = jnp.pi / 3  # 60°
SQUARE_ANGLE_RAD = jnp.pi / 2   # 90°
PENTAGON_ANGLE_RAD = 3 * jnp.pi / 5  # 108°


def test_angles_polygon_triangle():
    """Regular triangle: all internal angles are 60°."""
    polygon = Polygon.from_sides_and_radius_xy(3, 1.0)
    xyz = jnp.array(polygon.points)
    angles = angles_polygon(xyz, deg=False)
    expected = jnp.full(3, TRIANGLE_ANGLE_RAD)

    assert jnp.allclose(angles, expected)


def test_angles_polygon_square():
    """Regular square: all internal angles are 90°."""
    polygon = Polygon.from_sides_and_radius_xy(4, 1.0)
    xyz = jnp.array(polygon.points)
    angles = angles_polygon(xyz, deg=False)
    expected = jnp.full(4, SQUARE_ANGLE_RAD)

    assert jnp.allclose(angles, expected)


def test_angles_polygon_rectangle():
    """Rectangle: all internal angles are 90° regardless of aspect ratio."""
    # 2x1 rectangle
    rectangle = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    angles = angles_polygon(rectangle, deg=False)
    expected = jnp.full(4, SQUARE_ANGLE_RAD)

    assert jnp.allclose(angles, expected)


def test_angles_polygon_irregular_quad():
    """angles_polygon and cosines_angles_polygon handle irregular quadrilaterals."""
    # Irregular quad (trapezoid-like): non-equal sides, non-90° angles
    quad = jnp.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.5, 2.0, 0.0],
        [0.5, 1.5, 0.0],
    ])
    angles = angles_polygon(quad, deg=False)
    cosines = cosines_angles_polygon(quad)

    # Internal angles of a simple quad sum to 2π
    assert jnp.allclose(jnp.sum(angles), 2 * jnp.pi)
    # cosines match cos(angles)
    assert jnp.allclose(cosines, jnp.cos(angles))


def test_angles_polygon_pentagon():
    """Regular pentagon: all internal angles are 108°."""
    polygon = Polygon.from_sides_and_radius_xy(5, 1.0)
    xyz = jnp.array(polygon.points)
    angles = angles_polygon(xyz, deg=False)
    expected = jnp.full(5, PENTAGON_ANGLE_RAD)

    assert jnp.allclose(angles, expected)


def test_angles_polygon_degrees():
    """angles_polygon with deg=True returns degrees."""
    polygon = Polygon.from_sides_and_radius_xy(4, 1.0)
    xyz = jnp.array(polygon.points)
    angles_deg = angles_polygon(xyz, deg=True)
    angles_rad = angles_polygon(xyz, deg=False)

    assert jnp.allclose(angles_deg, jnp.degrees(angles_rad))


def test_angles_polygon_2d():
    """angles_polygon works with 2D polygons."""
    square_2d = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    angles = angles_polygon(square_2d, deg=False)
    expected = jnp.full(4, SQUARE_ANGLE_RAD)

    assert jnp.allclose(angles, expected)


def test_angles_polygon_batched():
    """angles_polygon works with batched polygons via vmap."""
    # Batch of pentagons with different radii (same vertex count)
    pentagons = jnp.stack([
        jnp.array(Polygon.from_sides_and_radius_xy(5, 1.0).points),
        jnp.array(Polygon.from_sides_and_radius_xy(5, 2.0).points),
        jnp.array(Polygon.from_sides_and_radius_xy(5, 0.5).points),
    ])
    batched_angles = vmap(angles_polygon, in_axes=(0, None))(pentagons, False)

    # All should have 108° internal angles (scale invariant)
    expected = jnp.full((3, 5), PENTAGON_ANGLE_RAD)
    assert jnp.allclose(batched_angles, expected)

    # Match sequential calls
    for i in range(3):
        single = angles_polygon(pentagons[i], deg=False)
        assert jnp.allclose(batched_angles[i], single)


def test_angles_polygon_triangle_padded_repeats():
    """angles_polygon handles triangle with last two vertices repeating the first."""
    triangle = Polygon.from_sides_and_radius_xy(3, 1.0)
    pts = jnp.array(triangle.points)
    xyz = jnp.vstack([pts, pts[0:1], pts[0:1]])  # 5 vertices; indices 3,4 = vertex 0

    angles = angles_polygon(xyz, deg=False)

    # Vertices 1-2 form proper triangle corners; each has internal angle 60°
    assert jnp.allclose(angles[1:3], jnp.full(2, TRIANGLE_ANGLE_RAD))


def test_cosines_angles_polygon_triangle_padded_repeats():
    """cosines_angles_polygon handles triangle with last two vertices repeating the first."""
    triangle = Polygon.from_sides_and_radius_xy(3, 1.0)
    pts = jnp.array(triangle.points)
    xyz = jnp.vstack([pts, pts[0:1], pts[0:1]])  # 5 vertices; indices 3,4 = vertex 0

    cosines = cosines_angles_polygon(xyz)

    # Vertices 1-2 form proper triangle corners; cos(60°) = 0.5
    assert jnp.allclose(cosines[1:3], jnp.full(2, TRIANGLE_COSINE))


def test_angles_polygon_quad_padded_repeats():
    """angles_polygon handles quad with last two vertices repeating the first."""
    quad = Polygon.from_sides_and_radius_xy(4, 1.0)
    pts = jnp.array(quad.points)
    xyz = jnp.vstack([pts, pts[0:1], pts[0:1]])  # 6 vertices; indices 4,5 = vertex 0

    angles = angles_polygon(xyz, deg=False)

    # Vertices 1-3 form proper quad corners; each has internal angle 90°
    assert jnp.allclose(angles[1:4], jnp.full(3, SQUARE_ANGLE_RAD))


def test_cosines_angles_polygon_quad_padded_repeats():
    """cosines_angles_polygon handles quad with last two vertices repeating the first."""
    quad = Polygon.from_sides_and_radius_xy(4, 1.0)
    pts = jnp.array(quad.points)
    xyz = jnp.vstack([pts, pts[0:1], pts[0:1]])  # 6 vertices; indices 4,5 = vertex 0

    cosines = cosines_angles_polygon(xyz)

    # Vertices 1-3 form proper quad corners; cos(90°) = 0
    # assert jnp.allclose(cosines[1:4], jnp.full(3, SQUARE_COSINE))
    assert jnp.allclose(cosines, jnp.full(6, SQUARE_COSINE))


def test_angles_polygon_batched_uniform():
    """angles_polygon batched over multiple squares (same vertex count)."""
    # Create 4 unit squares at different positions
    squares = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
        [[2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [2.0, 3.0, 0.0]],
    ])
    batched_angles = vmap(angles_polygon, in_axes=(0, None))(squares, True)
    expected = jnp.full((4, 4), 90.0)

    assert jnp.allclose(batched_angles, expected)


# Analytical cos(internal angle) for regular n-gons
TRIANGLE_COSINE = jnp.cos(TRIANGLE_ANGLE_RAD)   # cos(60°) = 0.5
SQUARE_COSINE = jnp.cos(SQUARE_ANGLE_RAD)       # cos(90°) = 0
PENTAGON_COSINE = jnp.cos(PENTAGON_ANGLE_RAD)  # cos(108°) ≈ -0.309


def test_cosines_angles_polygon_triangle():
    """Regular triangle: all cos(internal) = cos(60°) = 0.5."""
    polygon = Polygon.from_sides_and_radius_xy(3, 1.0)
    xyz = jnp.array(polygon.points)
    cosines = cosines_angles_polygon(xyz)
    expected = jnp.full(3, TRIANGLE_COSINE)

    assert jnp.allclose(cosines, expected)


def test_cosines_angles_polygon_square():
    """Regular square: all cos(internal) = cos(90°) = 0."""
    polygon = Polygon.from_sides_and_radius_xy(4, 1.0)
    xyz = jnp.array(polygon.points)
    cosines = cosines_angles_polygon(xyz)
    expected = jnp.full(4, SQUARE_COSINE)

    assert jnp.allclose(cosines, expected)


def test_cosines_angles_polygon_rectangle():
    """Rectangle: all cos(internal) = 0 regardless of aspect ratio."""
    rectangle = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    cosines = cosines_angles_polygon(rectangle)
    expected = jnp.full(4, SQUARE_COSINE)

    assert jnp.allclose(cosines, expected)


def test_cosines_angles_polygon_pentagon():
    """Regular pentagon: all cos(internal) = cos(108°) ≈ -0.309."""
    polygon = Polygon.from_sides_and_radius_xy(5, 1.0)
    xyz = jnp.array(polygon.points)
    cosines = cosines_angles_polygon(xyz)
    expected = jnp.full(5, PENTAGON_COSINE)

    assert jnp.allclose(cosines, expected)


def test_cosines_angles_polygon_matches_angles_polygon():
    """cosines_angles_polygon equals cos(angles_polygon)."""
    polygon = Polygon.from_sides_and_radius_xy(5, 1.0)
    xyz = jnp.array(polygon.points)
    cosines = cosines_angles_polygon(xyz)
    angles = angles_polygon(xyz, deg=False)

    assert jnp.allclose(cosines, jnp.cos(angles))


def test_cosines_angles_polygon_2d():
    """cosines_angles_polygon works with 2D polygons."""
    square_2d = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    cosines = cosines_angles_polygon(square_2d)
    expected = jnp.full(4, SQUARE_COSINE)

    assert jnp.allclose(cosines, expected)


def test_cosines_angles_polygon_batched():
    """cosines_angles_polygon works with batched polygons via vmap."""
    pentagons = jnp.stack([
        jnp.array(Polygon.from_sides_and_radius_xy(5, 1.0).points),
        jnp.array(Polygon.from_sides_and_radius_xy(5, 2.0).points),
    ])
    batched_cosines = vmap(cosines_angles_polygon)(pentagons)
    expected = jnp.full((2, 5), PENTAGON_COSINE)

    assert jnp.allclose(batched_cosines, expected)


def test_cosines_angles_polygon_mesh_faces_quad_triangles():
    """cosines_angles_polygon on inner faces of a meshgrid quad mesh."""
    mesh = FDMesh.from_meshgrid(10.0, 5)

    # Randomly rotate the mesh
    key = jax.random.PRNGKey(1701)
    angles = jax.random.uniform(key, (3,), minval=0.0, maxval=2 * jnp.pi)
    mesh = mesh.transformed(Rotation.from_euler_angles(angles))

    # split the faces into corners to build a mesh with quads and triangles
    # find the corner and opposite vertex of each face
    face_corners = {}
    for face in mesh.faces():
        if not mesh.is_face_on_boundary(face):
            continue
        if len(mesh.face_neighbors(face)) != 2:
            continue

        corner = None
        face_vertices = mesh.face_vertices(face)
        for index, vkey in enumerate(face_vertices):
            if len(mesh.vertex_neighborhood(vkey)) == 2:
                corner = vkey
                break
        if corner is None:
            raise Exception("Face has no corner")

        # find the opposite vertex
        hood = cycle(face_vertices)
        for _ in range(index + 3):
            opposite = next(hood)
        face_corners[face] = (corner, opposite)

    # split the corners faces into triangles
    for fkey, (corner, opposite) in face_corners.items():
        mesh.split_face(fkey, corner, opposite)

    # gather data
    xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
    structure = EquilibriumMeshStructure.from_mesh(mesh)

    # check the cosines of the inner quad faces
    for face in structure.faces_indexed:
        # Quad faces: use only valid vertices (no -1 padding for uniform quad mesh)
        face_valid = face[face >= 0]
        fxyz = xyz[face_valid, :]
        if len(fxyz) != 4:
            continue
        cosines = cosines_angles_polygon(fxyz)
        # Inner quad faces are rectangles: all internal angles 90°, cos(90°) = 0
        assert jnp.allclose(cosines, 0.0)

