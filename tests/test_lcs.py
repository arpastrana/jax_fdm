from jax import jit
import jax.numpy as jnp

from compas.geometry import Polygon
from compas.geometry import Rotation
from compas.geometry import Frame
from compas.geometry import Vector

from jax_fdm.geometry import WORLD_XYZ
from jax_fdm.geometry import WORLD_X
from jax_fdm.geometry import WORLD_Y
from jax_fdm.geometry import WORLD_Z
from jax_fdm.geometry import line_lcs
from jax_fdm.geometry import polygon_lcs


load = Vector(0.0, 0.0, 1.0)
load_scale = 2.0
polygon = Polygon.from_sides_and_radius_xy(4, 1.0)


def test_lcs_polygon_world_xyz():
    """
    Test the LCS of a polygon aligned with the world XYZ coordinate system.
    """
    lcs = polygon_lcs(jnp.array(polygon.points))
    assert jnp.allclose(lcs, WORLD_XYZ), f"lcs:\n{lcs}\nlcs_target:\n{WORLD_XYZ}"

    load_lcs = jnp.array(load * load_scale) @ lcs
    load_target = WORLD_Z * load_scale
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"

    load_lcs = -jnp.array(load * load_scale) @ lcs
    load_target = load_target * -1.0
    assert jnp.allclose(load_lcs, load_target), f"lcs:\n{load_lcs}\nlcs_target:\n{load_target}"


def test_lcs_polygon_world_zx():
    """
    Test the LCS of a polygon aligned with the world ZX coordinate system.
    """
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


def test_lcs_polygon_world_yz():
    """
    Test the LCS of a polygon aligned with the world YZ coordinate system.
    """
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


def test_lcs_line_on_x():
    """
    Test the LCS of a line parallel to the world X axis.
    """
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    line = jnp.array(line)
    lcs = line_lcs(line)
    lcs_target = WORLD_XYZ
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"


def test_lcs_line_on_y():
    """
    Test the LCS of a line parallel to the world Y axis.
    """
    line = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    line = jnp.array(line)
    lcs = line_lcs(line)

    lcs_target = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"


def test_lcs_line_on_z():
    """
    Test the LCS of a line parallel to the world Z axis.
    """
    line = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    line = jnp.array(line)
    lcs = line_lcs(line)

    lcs_target = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert jnp.allclose(lcs, lcs_target), f"lcs:\n{lcs}\nlcs_target:\n{lcs_target}"
