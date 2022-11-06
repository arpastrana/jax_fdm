"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os

# math
from math import radians
from math import sqrt

# compas
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.utilities import pairwise

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import EdgeDirectionGoal

from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss

from jax_fdm.optimization import LBFGSB
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
num_sides = 8
num_rings = 40
offset_distance = 0.01  # ring offset

# initial form-finding parameters
q0_ring = -2.0  # starting force density for ring (hoop) edges
q0_cross = -0.5  # starting force density for the edges transversal to the rings
pz = -0.1  # z component of the applied load
qmin, qmax = None, None

# optimization
optimizer = LBFGSB
maxiter = 10000
tol = 1e-6  # 1e-6 for best results at the cost of a considerable speed decrease

# goal length
length_target = 0.03

# goal vector, angle
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_base = 20.0  # angle constraint, lower bound
angle_top = 30.0  # angle constraint, upper bound

# io
record = True
export = False

HERE = os.path.dirname(__file__)

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Create the base geometry of the dome
# ==========================================================================

polygon = Polygon.from_sides_and_radius_xy(num_sides, diameter / 2.).points

rings = []
for i in range(num_rings + 1):
    polygon = offset_polygon(polygon, offset_distance)
    nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in polygon]
    rings.append(nodes)

edges_rings = []
for ring in rings[1:]:
    for u, v in pairwise(ring + ring[:1]):
        edge = network.add_edge(u, v)
        edges_rings.append(edge)

crosses = []
edges_cross = []
for i in range(num_sides):

    radial = []
    for ring in rings:
        radial.append(ring[i])
    crosses.append(radial)

    for u, v in pairwise(radial):
        edge = network.add_edge(u, v)
        edges_cross.append(edge)

edges_cross_rings = []
for rings_pair in pairwise(rings):
    cross_ring = []
    for edge in zip(*rings_pair):
        cross_ring.append(edge)
    edges_cross_rings.append(cross_ring)

# ==========================================================================
# Define structural system
# ==========================================================================

# define anchors
for node in rings[0]:
    network.node_anchor(node)

# apply loads to unanchored nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all edges
q0_scale = sqrt(network.number_of_edges()) / 2.

for edge in edges_rings:
    network.edge_forcedensity(edge, q0_ring)

for i, cross_ring in enumerate(edges_cross_rings):
    for edge in cross_ring:
        network.edge_forcedensity(edge, q0_cross * q0_scale * (num_rings - i))

# ==========================================================================
# Store network
# ==========================================================================

networks = {"start": network}

if export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_base.json")
    network.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

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

goals = []

# edge length goal
for cross_ring in edges_cross_rings:
    for edge in cross_ring:
        goal = EdgeLengthGoal(edge, target=length_target, weight=1.)
        goals.append(goal)

# edge vector goal
vectors_goal = []
for i, cross_ring in enumerate(edges_cross_rings):

    angle_delta = angle_top - angle_base
    angle = angle_base + angle_delta * (i / (num_rings - 1))

    print(f"Edges ring {i + 1}/{len(edges_cross_rings)}. Angle goal: {angle}")

    for u, v in cross_ring:

        edge = (u, v)
        xyz = network.node_coordinates(u)  # xyz of first node, assumes it is the lowermost
        normal = cross_vectors(network.edge_vector(u, v), angle_vector)

        point = add_vectors(xyz, angle_vector)
        end = rotate_points([point], -radians(angle), axis=normal, origin=xyz).pop()
        vector = subtract_vectors(end, xyz)

        goal = EdgeDirectionGoal(edge, target=vector, weight=1.0)
        goals.append(goal)
        vectors_goal.append((vector, edge))

# ==========================================================================
# Define loss function with goals
# ==========================================================================

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Form-finding sweep
# ==========================================================================

sweep_configs = [{"name": "eq",
                  "method": fdm,
                  "msg": "\n*Form found network*",
                  "save": True},
                 {"name": "eq_g",
                 "method": constrained_fdm,
                  "msg": "\n*Constrained form found network. No constraints*",
                  "save": True,
                  "record": record},
                 ]

# ==========================================================================
# Print out stats
# ==========================================================================

for config in sweep_configs:

    fofin_method = config["method"]

    print()
    print(config["msg"])

    if fofin_method == fdm:
        network = fofin_method(network)
    else:
        optimizer = optimizer()

        recorder = OptimizationRecorder(optimizer) if config.get("record") else None

        network = fofin_method(network,
                               optimizer=optimizer,
                               parameters=parameters,
                               loss=loss,
                               constraints=config.get("constraints", []),
                               maxiter=maxiter,
                               tol=tol,
                               callback=recorder)

    # store network
    if config["save"]:
        networks[config["name"]] = network

    # Report stats
    network.print_stats()

# ==========================================================================
# Export optimization history
# ==========================================================================

    if record and export and fofin_method != fdm:
        FILE_OUT = os.path.join(HERE, f"../data/json/{name}_history.json")
        recorder.to_json(FILE_OUT)
        print("Optimization history exported to", FILE_OUT)

# ==========================================================================
# Plot loss components
# ==========================================================================

    if record and fofin_method != fdm:
        plotter = LossPlotter(loss, network, dpi=150, figsize=(8, 4))
        plotter.plot(recorder.history)
        plotter.show()

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    model_name = config["name"]
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_{model_name}_optimized.json")
    network.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

# optimized network
c_network = networks["eq_g"]
viewer.add(c_network, edgewidth=(0.003, 0.03), edgecolor="force", reactionscale=0.1)

# add target vectors
for vector, edge in vectors_goal:
    u, v = edge
    xyz = c_network.node_coordinates(u)
    viewer.add(Line(xyz, add_vectors(xyz, scale_vector(vector, 0.1))))

# show le crème
viewer.show()
