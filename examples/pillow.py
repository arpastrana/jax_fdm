"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os

from math import fabs

from random import random

import numpy as np

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector

# quads
from compas_singular.datastructures import CoarseQuadMesh

# visualization
from compas_view2.app import App

# jax_fdm
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodeLineGoal
from jax_fdm.goals import NetworkLoadPathGoal

from jax_fdm.constraints import NodeCurvatureConstraint
from jax_fdm.constraints import NetworkEdgesLengthConstraint
from jax_fdm.constraints import NetworkEdgesForceConstraint

from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss

from jax_fdm.optimization import SLSQP


# ==========================================================================
# Initial parameters
# ==========================================================================

model_name = "pillow"

# geometric parameters
l1 = 10.0
l2 = 10.0
divisions = 10

# initial form-finding parameters
q0, dq = -2.0, 0.1  # starting average force density and random deviation
pz = -100.0  # z component of the total applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-3

# parameter bounds
qmin = None
qmax = None

# goal horizontal projection
add_horizontal_projection_goal = True
weight_horizontal_projection = 1.0

# goal load path
add_load_path_goal = False
normalise_by_edge_number = False
weight_load_path = 0.001

# goal edge length
add_edge_length_goal = False
weight_edge_length = 1.0

# constraint length
add_edge_length_constraint = True
ratio_length_min = 0.5
ratio_length_max = 3.0

# constraint force
add_edge_force_constraint = True
force_min = -100.0
force_max = -1.0

# constraint curvature
add_curvature_constraint = True
crv_min = -100.0
crv_max = -0.1

export = False

# ==========================================================================
# Create base geometry
# ==========================================================================

vertices = [[l1, 0.0, 0.0], [l1, l2, 0.0], [0.0, l2, 0.0], [0.0, 0.0, 0.0]]
faces = [[0, 1, 2, 3]]
# faces = [[3, 2, 1, 0]]
coarse = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)

coarse.collect_strips()
coarse.set_strips_density(divisions)
coarse.densification()
mesh = coarse.get_quad_mesh()

vertices, _ = mesh.to_vertices_and_faces()
network = FDNetwork.from_nodes_and_edges(vertices, mesh.edges())

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for key in network.nodes():
    if mesh.is_vertex_on_boundary(key):
        network.node_support(key)

# set initial q to all edges
for edge in network.edges():
    q = q0 + dq * (random() - 0.5)
    network.edge_forcedensity(edge, q)

networks = {"input": network}

# ==========================================================================
# Initial form finding - no external loads
# ==========================================================================

networks["unloaded"] = fdm(network)

# ==========================================================================
# Initial form finding - loaded
# ==========================================================================

# apply loads
mesh_area = mesh.area()
for key in network.nodes():
    network.node_load(key, load=[0.0, 0.0, pz * mesh.vertex_area(key) / mesh_area])

networks["loaded"] = fdm(network)

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# horizontal projection goal
if add_horizontal_projection_goal:
    print("Horizontal projection goal")
    for node in network.nodes_free():
        xyz = network.node_coordinates(node)
        line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
        goal = NodeLineGoal(node, target=line, weight=weight_horizontal_projection)
        goals.append(goal)

# load path goal
if add_load_path_goal:
    if normalise_by_edge_number:
        weight_load_path /= mesh.number_of_edges()
    goals.append(NetworkLoadPathGoal(target=0.0, weight=weight_load_path))

# edge length goal
if add_edge_length_goal:
    network2 = networks["loaded"]
    for edge in network.edges():
        goal = EdgeLengthGoal(edge, network2.edge_length(*edge), weight=weight_edge_length)
        goals.append(goal)

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create constraints
# ==========================================================================

constraints = []

if add_edge_length_constraint:
    average_length = np.mean([network.edge_length(*edge) for edge in network.edges()])
    length_min = ratio_length_min * average_length
    length_max = ratio_length_max * average_length
    constraint = NetworkEdgesLengthConstraint(bound_low=length_min,
                                              bound_up=length_max)
    constraints.append(constraint)

    msg = "Edge length constraint between {} and {}"
    print(msg.format(round(length_min, 2), round(length_max, 2)))

if add_edge_force_constraint:
    constraint = NetworkEdgesForceConstraint(bound_low=force_min,
                                             bound_up=force_max)
    constraints.append(constraint)

    msg = "Edge force constraint between {} and {}"
    print(msg.format(round(force_min, 2), round(force_max, 2)))

if add_curvature_constraint:
    polyedge0 = mesh.collect_polyedge(*mesh.edges_on_boundary()[0])
    n = len(polyedge0)
    i = int(n / 2)
    u0, v0 = polyedge0[i - 1: i + 1]

    if mesh.halfedge[u0][v0] is None:
        u0, v0 = v0, u0

    u, v = mesh.halfedge_after(u0, v0)
    polyedge = mesh.collect_polyedge(u, v)
    subpolyedge = polyedge[1:-1]

    for key in subpolyedge:
        polygon = mesh.vertex_neighbors(key, ordered=True)
        constraint = NodeCurvatureConstraint(key,
                                             polygon,
                                             bound_low=crv_min,
                                             bound_up=crv_max)
        constraints.append(constraint)

    msg = "Node curvature constraint between {} and {} on {} nodes"
    print(msg.format(round(crv_min, 2), round(crv_max, 2), len(subpolyedge)))

# ==========================================================================
# Form finding
# ==========================================================================

networks["free"] = fdm(network)

networks["uncstr_opt"] = constrained_fdm(network,
                                         optimizer=optimizer(),
                                         bounds=(qmin, qmax),
                                         loss=loss,
                                         maxiter=maxiter)

networks["cstr_opt"] = constrained_fdm(network,
                                       optimizer=optimizer(),
                                       bounds=(qmin, qmax),
                                       loss=loss,
                                       constraints=constraints,
                                       maxiter=maxiter)

# ==========================================================================
# Print and export results
# ==========================================================================

for network_name, network in networks.items():

    print()
    print("Design {}".format(network_name))

    print(f"Load path: {round(network.loadpath(), 3)}")

    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    data = {"Force densities": q, "Forces": f, "Lengths": l}

    for name, values in data.items():
        minv = round(min(values), 3)
        maxv = round(max(values), 3)
        meanv = round(np.mean(values), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    if export:
        HERE = os.path.dirname(__file__)
        FILE_OUT = os.path.join(HERE, "../data/json/{}_{}.json".format(model_name, network_name))
        network.to_json(FILE_OUT)
        print("Design {} exported to".format(network_name), FILE_OUT)

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# add all networks except the last one
networks = list(networks.values())
for i, network in enumerate(networks):
    if i == (len(networks) - 1):
        continue
    viewer.add(network, show_points=False, linewidth=1.0, color=Color.grey().darkened(i * 10))

network0 = networks[0]
if len(networks) > 1:
    c_network = networks[-1]  # last network is colored
else:
    c_network = networks[0]

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
           pointsize=10.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in c_network.nodes():

    pt = c_network.node_coordinates(node)

    # draw residual forces
    residual = c_network.node_residual(node)

    if length_vector(residual) < 0.001:
        continue

    # print(node, residual, length_vector(residual))
    residual_line = Line(pt, add_vectors(pt, residual))
    viewer.add(residual_line,
               linewidth=4.0,
               color=Color.pink())  # Color.purple()

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
