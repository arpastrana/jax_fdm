"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os
import matplotlib.pyplot as plt

# math
from math import fabs
from math import radians
from math import sqrt

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.utilities import pairwise

# visualization
from compas_view2.app import App

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import EdgeDirectionGoal
from jax_fdm.goals import EdgeAngleGoal

from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss

from jax_fdm.optimization import BFGS

from jax_fdm.optimization import OptimizationRecorder

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

# optimization
optimizer = BFGS
maxiter = 10000
tol = 1e-3  # 1e-6 for best results at the cost of a considerable speed decrease

# parameter bounds
qmin = None  # -200.0
qmax = None  # -0.001

# goal length
length_target = 0.03

# goal vector, angle
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_base = 20.0  # angle constraint, lower bound
angle_top = 30.0  # angle constraint, upper bound

# io
export = False
record = False

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

# define supports
for node in rings[0]:
    network.node_support(node)

# apply loads to unsupported nodes
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
# Create goals
# ==========================================================================

goals = []

# edge length goal
for cross_ring in edges_cross_rings:
    for edge in cross_ring:
        goal = EdgeLengthGoal(edge, target=length_target, weight=1.)
        goals.append(goal)

# edge vector goal
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
        recorder = None
        if config.get("record"):
            recorder = OptimizationRecorder()

        network = fofin_method(network,
                               optimizer=optimizer(),
                               bounds=(qmin, qmax),
                               loss=loss,
                               constraints=config.get("constraints", []),
                               maxiter=maxiter,
                               tol=tol,
                               callback=recorder)

    # store network
    if config["save"]:
        networks[config["name"]] = network

    # Report stats
    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    fields = [q, f, l]
    field_names = ["FDs", "Forces", "Lengths"]

    print(f"Load path: {round(network.loadpath(), 3)}")
    for field_name, vals in zip(field_names, fields):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{field_name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

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
        model = EquilibriumModel(network)
        fig = plt.figure(dpi=150)
        for loss_term in [loss] + list(loss.terms):
            y = []
            for q in recorder.history:
                eqstate = model(q)
                try:
                    error = loss_term(eqstate)
                except:
                    error = loss_term(q, model)
                y.append(error)
            plt.plot(y, label=loss_term.name)

        plt.xlabel("Optimization iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.show()

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

viewer = App(width=1600, height=900, show_grid=False)

# add all networks except the last one
networks = list(networks.values())

# for i, network in enumerate(networks):
#     if i == (len(networks) - 1):
#         continue
#     viewer.add(network, show_points=False, linewidth=1.0, color=Color.grey().darkened(i * 10))

network0 = networks[0]
if len(networks) > 1:
    c_network = networks[-1]  # last network is colored
else:
    c_network = networks[0]

# for vector, edge in vector_edges:
#     u, v = edge
#     xyz = c_network.node_coordinates(u)
#     viewer.add(Line(xyz, add_vectors(xyz, scale_vector(vector, 0.1))))

# plot the last network
# edges color map
cmap = ColorMap.from_mpl("viridis")

fds = [fabs(c_network.edge_forcedensity(edge)) for edge in c_network.edges()]
colors = {}
for edge in c_network.edges():
    fd = fabs(c_network.edge_forcedensity(edge))
    try:
        ratio = (fd - min(fds)) / (max(fds) - min(fds))
    except ZeroDivisionError:
        ratio = 1.
    colors[edge] = cmap(ratio)

# optimized network
viewer.add(c_network,
           show_vertices=True,
           pointsize=20.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in c_network.nodes():

    pt = c_network.node_coordinates(node)

    # draw lines betwen subject and target nodes
    # target_pt = network0.node_coordinates(node)
    # viewer.add(Line(target_pt, pt), linewidth=1.0, color=Color.grey().lightened())

    # draw residual forces
    residual = c_network.node_residual(node)

    if length_vector(residual) < 0.001:
        continue

    # print(node, residual, length_vector(residual))
    # residual_line = Line(pt, add_vectors(pt, residual))
    # viewer.add(residual_line,
    #            linewidth=4.0,
    #            color=Color.pink())

# draw applied loads
for node in c_network.nodes():
    pt = c_network.node_coordinates(node)
    load = c_network.node_load(node)
    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=4.0,
               color=Color.green().darkened())

# draw supports
for node in c_network.nodes_supports():
    x, y, z = c_network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=30)

# show le crème
viewer.show()
