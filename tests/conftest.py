"""
Shared fixtures and helpers for the characterization test suite.
"""

import json
import os

import jax.numpy as jnp
import pytest

from compas.geometry import Polyline
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork

HERE = os.path.dirname(__file__)
BASELINES = os.path.join(HERE, "baselines")
DATA = os.path.join(HERE, "data")

# Set JFDM_CAPTURE_BASELINES=1 on the reference (COMPAS 1.x) env to (re)write the
# golden files, then run without it so they become assertions.
CAPTURE = os.environ.get("JFDM_CAPTURE_BASELINES") == "1"


def assert_baseline(name, value):
    """
    Assert that value matches golden <name>.json, or write it when capturing.

    The value must be JSON-serializable (use nested lists, not arrays).
    Returns the baseline so callers may reuse it.
    """
    path = os.path.join(BASELINES, f"{name}.json")
    if CAPTURE:
        os.makedirs(BASELINES, exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, indent=2)
        return value

    with open(path) as f:
        baseline = json.load(f)
    assert jnp.allclose(jnp.asarray(value), jnp.asarray(baseline), rtol=1e-6, atol=1e-9)

    return baseline


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def arch_network():
    """
    A 10-segment line arch, anchored at both ends, with uniform force density
    and a vertical point load on every free node.
    """
    num_segments = 10
    start = [-2.5, 0.0, 0.0]
    end = [2.5, 0.0, 0.0]
    points = Polyline([start, end]).divide_polyline(num_segments)
    lines = Polyline(points).lines

    network = FDNetwork.from_lines(lines)
    network.node_anchor(key=0)
    network.node_anchor(key=len(points) - 1)
    network.edges_forcedensities(-1.0, keys=network.edges())
    network.nodes_loads([0.0, 0.0, -0.2], keys=network.nodes_free())

    return network


@pytest.fixture
def meshgrid_mesh():
    """
    A regular quad meshgrid.
    """
    return FDMesh.from_meshgrid(dx=2, nx=5)


@pytest.fixture
def meshgrid_mesh_fine():
    """
    A finer regular quad meshgrid.
    """
    return FDMesh.from_meshgrid(dx=2, nx=9)


@pytest.fixture
def tetra_mesh():
    """
    A tetrahedron mesh with triangular faces.
    """
    return FDMesh.from_polyhedron(4)


@pytest.fixture
def octa_mesh():
    """
    An octahedron mesh with triangular faces.
    """
    return FDMesh.from_polyhedron(8)


@pytest.fixture
def ragged_mesh():
    """
    A quad meshgrid with two internal faces split into triangles.

    The result mixes quad and triangular faces, so its `faces_indexed` rows have
    unequal valid lengths and the shorter (triangular) rows are `-1`-padded. This
    is what drives the padding path in the vertex-normal machinery — the all-quad
    `meshgrid_mesh` never does.
    """
    mesh = FDMesh.from_meshgrid(dx=2, nx=5)

    internal = [fkey for fkey in mesh.faces() if not mesh.is_face_on_boundary(fkey)]
    for fkey in internal[:2]:
        vertices = mesh.face_vertices(fkey)
        # split along a diagonal to turn one quad into two triangles
        mesh.split_face(fkey, vertices[0], vertices[2])

    return mesh


@pytest.fixture
def irregular_mesh():
    """
    An irregular, non-planar quad mesh used as a pillow-fixture substitute.
    """
    return FDMesh.from_json(os.path.join(DATA, "irregular_mesh.json"))
