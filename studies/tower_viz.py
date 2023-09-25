from math import cos
from math import sin
from math import pi
from math import radians

from compas.geometry import Polygon

from compas.utilities import geometric_key
from compas.utilities import pairwise
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from functools import partial
import time

import pickle
import jax


# ==========================================================================
# Create the geometry of a tower
# ==========================================================================

def polygon_from_sides_and_radius_xy(num_segments, radius, z):
    """
    """
    points = []
    for i in range(num_segments):
        angle = 2 * pi * i / num_segments
        x = radius * cos(angle)
        y = radius * sin(angle)
        points.append([x, y, z])

    return Polygon(points)


def create_tower_rings(radius_bottom, radius_top, height, num_segments, num_rings):
    """
    """
    rings = []

    ring_height = height / (num_rings - 1)
    radii = remap_values(list(range(num_rings)), radius_bottom, radius_top)

    for i in range(num_rings):

        z = i * ring_height
        radius = radii[i]

        ring = polygon_from_sides_and_radius_xy(num_segments, radius, z)

        # get edge midpoint for odd rings
        if i > 0 and i % 2 != 0:
            points = [line.midpoint for line in ring.lines]
            ring = Polygon(points)

        rings.append(ring)

    return rings


def triangulate_rings(rings):
    """
    """
    lines = []

    for i, rings_pair in enumerate(pairwise(rings)):
        for line in zip(*rings_pair):
            lines.append(line)

        if i > 0 and i % 2 != 0:
            ring_b, ring_a = rings_pair
        else:
            ring_a, ring_b = rings_pair

        ring_a_points = ring_a.points
        ring_a = ring_a_points[1:] + ring_a_points[:1]
        rings_pair = (ring_a, ring_b)

        for line in zip(*rings_pair):
            lines.append(line)

    return lines


def create_tower_geometry(num_segments, radius_bottom, radius_top, height, num_rings):
    """
    """
    rings = create_tower_rings(num_segments, radius_bottom, radius_top, height, num_rings)
    diagonals = triangulate_rings(rings)

    return rings, diagonals


def create_tower_network(rings, diagonals, q, ratio):
    """
    """
    network = FDNetwork()

    # add nodes
    q_ring = q
    for i, ring in enumerate(rings):
        is_first_or_last = (i == 0 or i == len(rings) - 1)

        ring_nodes = []
        for point in ring.points:
            node = network.add_node(attr_dict={k: v for k, v in zip("xyz", point)})
            ring_nodes.append(node)

            if is_first_or_last:
                network.node_support(node)

        # add rings edges if ring is not first or last
        if not is_first_or_last:
            for u, v in pairwise(ring_nodes + ring_nodes[:1]):
                network.add_edge(u, v, attr_dict={"q": q_ring})

    # add diagonal edges
    gkey_key = network.gkey_key()
    q_diag = q / ratio
    for line in diagonals:
        u, v = (gkey_key[geometric_key(point)] for point in line)

        if network.has_edge(u, v):
            print("has it")
            continue
        network.add_edge(u, v, attr_dict={"q": q_diag})

    return network


# ==========================================================================
# Main script
# ==========================================================================

if __name__ == "__main__":

    from compas.geometry import Translation
    from compas.geometry import Rotation

    from jax_fdm.equilibrium import fdm
    from jax_fdm.visualization import Plotter
    from jax_fdm.visualization import Viewer

    # experiment settings
    num_reps = 1  # 5
    num_segments = [4, 8, 16, 32, 64][:4]
    num_segments = [i for i in range(4, 21) if i % 2 == 0]
    num_segments = [10]

    # script parameters
    radius_bottom = 10.0
    radius_top = 2.0
    height = 5.0
    q_val = 1.
    ratio_radial_vertical = 0.5

    # viz controls
    use_viewer = False
    plot_save = True
    filepath = "tower.pdf"
    viz_options = {"edgecolor": "force",
                   "show_loads": False}

    # instantiate a plotter (only for visualization, optional)
    plotter = Plotter(figsize=(8, 5), dpi=300)

    if use_viewer:
        viewer = Viewer(width=1600, height=900, show_grid=False)

    # generate saddles of increasing number of side segments
    info = []

    for i, num_segments in enumerate(num_segments):

        assert num_segments % 2 == 0, "Only even number of segments"

        # create network
        num_rings = num_segments
        rings, diagonals = create_tower_geometry(radius_bottom,
                                                 radius_top,
                                                 height,
                                                 num_segments,
                                                 num_rings)
        network = create_tower_network(rings, diagonals, q_val, ratio_radial_vertical)

        print(f"Num segments:{num_segments}\t# nodes: {network.number_of_nodes()}")

        network_eq = fdm(network)

        # add network in equilibrium to plotter
        T = Translation.from_vector([i * (radius_bottom * 2 + 2.0), 0., 0.0])
        R = Rotation.from_euler_angles((radians(-60.0), 0.0, 0.0))
        plotter.add(network_eq.transformed(R).transformed(T),
                    show_reactions=False,
                    edgewidth=(0.4, 3.0),  # (0.12, 1.2)
                    show_nodes=True,
                    nodesize=12,
                    **viz_options)

        if use_viewer:
            network_eq = network_eq.transformed(T)
            viewer.add(network_eq,
                       show_reactions=True,
                       reactionscale=0.3,
                       edgewidth=(0.01, 0.1),
                       **viz_options)

    # save visualization plot
    if use_viewer:
        viewer.show()

    plotter.zoom_extents()
    if plot_save:
        print("Saving")
        plotter.save(filepath, dpi=300, bbox_inches="tight", transparent=True)
    plotter.show()
