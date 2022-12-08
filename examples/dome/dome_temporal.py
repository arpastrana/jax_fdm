"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os
from itertools import cycle


# math
from math import radians
from math import sqrt

# compas
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.geometry import Translation
from compas.utilities import pairwise

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import EdgeDirectionGoal
from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss

from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import TrustRegionConstrained
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Initial parameters
# ==========================================================================

name = "dome"

# geometric parameters
diameter = 1.0
num_sides = 16
num_rings = 4
offset_distance = 0.05  # ring offset in 2D

# initial form-finding parameters
q0_ring = -2.0  # starting force density for ring (hoop) edges
q0_cross = -0.5  # starting force density for the edges transversal to the rings
pz = -0.1  # z component of the applied load
qmin = None
qmax = None   # -0.01

# goal length
length_target = 0.1

# goal vector, angle
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute target angle
angle_base = 20.0
angle_top = 40.0
angle_linear_range = True

# optimization
optimizer = LBFGSB
maxiter = 50000
tol = 1e-12  # 1e-6 for best results at the cost of a considerable speed decrease

optimize_twice = False
optimizer_2 = TrustRegionConstrained
maxiter_2 = 1000
tol_2 = 1e-9

# io
record = False
export = False
view = True

HERE = os.path.dirname(__file__)

# ==========================================================================
# Helper functions
# ==========================================================================

def add_ring(network, polygon):
    nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in polygon]
    edges = [network.add_edge(u, v) for u, v in pairwise(nodes + nodes[:1])]
    return nodes, edges

def connect_rings(network, ring_a, ring_b):
    edges = []
    for u, v in zip(ring_a, ring_b):
        edge = network.add_edge(u, v)
        edges.append(edge)
    return edges

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

vectors_goal = []
angles_iter = cycle((angle_base, angle_top))
angle_delta = angle_top - angle_base

# ==========================================================================
# Create the base geometry of the dome
# ==========================================================================

polygon = Polygon.from_sides_and_radius_xy(num_sides, diameter / 2.).points

# add first ring
ring_nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in polygon]
# ring_nodes, ring_edges = add_ring(network, polygon)
# for edge in ring_edges:
    # network.edge_forcedensity(edge, q0_ring)

# set anchors
for node in ring_nodes:
    network.node_anchor(node)

networks = {0: network}

# create rings
for i in range(1, num_rings + 1):

    polygon = offset_polygon(polygon, offset_distance)
    ring_nodes_new, ring_edges = add_ring(network, polygon)
    radial_edges = connect_rings(network, ring_nodes, ring_nodes_new)
    ring_nodes = ring_nodes_new

# ==========================================================================
# Define structural system
# ==========================================================================

    for edge in ring_edges:
        network.edge_forcedensity(edge, q0_ring)

    for edge in radial_edges:
        network.edge_forcedensity(edge, q0_cross)

    for node in ring_nodes:
        network.node_load(node, load=[0.0, 0.0, pz])

# ==========================================================================
# Define parameters
# ==========================================================================

    parameters = []
    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

# ==========================================================================
# Create goals
# ==========================================================================

    # node xyz goal
    goals_point = []
    for node in network.nodes_free():
        if node in ring_nodes_new:
            continue
        point = network.node_coordinates(node)
        goal = NodePointGoal(node, point, weight=1.)
        goals_point.append(goal)

    # edge length goal
    goals_length = []
    for edge in radial_edges:
        goal = EdgeLengthGoal(edge, target=length_target, weight=1.)
        goals_length.append(goal)

    # edge direction goal
    if angle_linear_range:
        angle = angle_base + angle_delta * ((i - 1) / (num_rings - 1))
    else:
        angle = next(angles_iter)
    print(f"\nEdges ring {i} /{num_rings}. Angle goal: {angle:.2f}")

    goals_vector = []
    for u, v in radial_edges:

        xu, yu, _ = network.node_coordinates(u)  # xyz of first node, assumes it is the lowermost
        u_xyz = [xu, yu, 0.0]

        xv, yv, _ = network.node_coordinates(v)  # xyz of first node, assumes it is the lowermost
        v_xyz = [xv, yv, 0.0]

        point = add_vectors(u_xyz, angle_vector)
        normal = cross_vectors(subtract_vectors(v_xyz, u_xyz), angle_vector)
        end = rotate_points([point], -radians(angle), axis=normal, origin=u_xyz).pop()
        vector = subtract_vectors(end, u_xyz)

        edge = (u, v)
        goal = EdgeDirectionGoal(edge, target=vector, weight=1.)
        goals_vector.append(goal)
        vectors_goal.append((vector, edge))

# ==========================================================================
# Define loss function with goals
# ==========================================================================

    loss = Loss(SquaredError(goals=goals_length),
                SquaredError(goals=goals_vector),
                SquaredError(goals=goals_point))

# ==========================================================================
# Constrained form-finding
# ==========================================================================

    opt = optimizer()

    recorder = OptimizationRecorder(opt) if record else None

    network = constrained_fdm(network,
                              optimizer=opt,
                              parameters=parameters,
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol,
                              callback=recorder)

    if optimize_twice:
        network = constrained_fdm(network,
                                  optimizer=optimizer_2(),
                                  parameters=parameters,
                                  loss=loss,
                                  maxiter=maxiter_2,
                                  tol=tol_2,
                                  callback=recorder)
    # Report stats
    network.print_stats()

    lengths = []
    for edge in radial_edges:
        lengths.append(network.edge_length(*edge))
    print(f"Average edge length: {sum(lengths) / len(lengths) }")

# ==========================================================================
# Store network
# ==========================================================================

    if export:
        FILE_OUT = os.path.join(HERE, f"../data/json/{name}_{i}.json")
        network.to_json(FILE_OUT)
        print("Problem definition exported to", FILE_OUT)

    # store network
    networks[i] = network.copy()

# ==========================================================================
# Export optimization history
# ==========================================================================

    # if record and export and fofin_method != fdm:
    #     FILE_OUT = os.path.join(HERE, f"../data/json/{name}_history.json")
    #     recorder.to_json(FILE_OUT)
    #     print("Optimization history exported to", FILE_OUT)

# ==========================================================================
# Plot loss components
# ==========================================================================

    # if record and fofin_method != fdm:
    #     plotter = LossPlotter(loss, network, dpi=150, figsize=(8, 4))
    #     plotter.plot(recorder.history)
    #     plotter.show()

# ==========================================================================
# Export JSON
# ==========================================================================

# if export:
#     model_name = config["name"]
#     FILE_OUT = os.path.join(HERE, f"../data/json/{name}_{model_name}_optimized.json")
#     network.to_json(FILE_OUT)
#     print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Visualization
# ==========================================================================

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    # optimized network
    for i, network in networks.items():
        if i == 0:
            continue
        T = Translation.from_vector([i * 1.2, 0.0, 0.0])
        c_network = network.transformed(T)
        viewer.add(c_network,
                   edgewidth=(0.003, 0.02),
                   edgecolor="force",
                   reactionscale=0.25,
                   loadscale=0.5)

        # mesh = Mesh.from_lines([network.edge_coordinates(*edge) for edge in network.edges()],
        #                        delete_boundary_face=True)
        # mesh = mesh.transformed(T)
        # viewer.add(mesh, show_points=False, show_lines=False, opacity=0.5)

        # # add target vectors
        # for vector, edge in vectors_goal:
        #     u, v = edge
        #     xyz = c_network.node_coordinates(u)
        #     viewer.add(Line(xyz, add_vectors(xyz, scale_vector(vector, 0.05))))

    # show le crème
    viewer.show()
