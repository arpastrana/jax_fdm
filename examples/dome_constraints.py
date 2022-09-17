"""
Solve a constrained force density problem using gradient-based optimization.
"""
import numpy as np
from math import fabs
from math import radians

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
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.utilities import pairwise

# visualization
from compas_view2.app import App

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import EquilibriumModel

from jax_fdm.goals import EdgeVectorAngleGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodeLineGoal

from jax_fdm.constraints import EdgeVectorAngleConstraint
from jax_fdm.constraints import NetworkEdgesLengthConstraint
from jax_fdm.constraints import NetworkEdgesForceConstraint

from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import OptimizationRecorder


# ==========================================================================
# Initial parameters
# ==========================================================================

name = "dome"

# geometric parameters
diameter = 1.0
num_sides = 16
num_rings = 6
offset_distance = 0.03  # ring offset

# initial form-finding parameters
q0_ring = -2.0  # starting force density for ring (hoop) edges
q0_cross = -0.5  # starting force density for the edges transversal to the rings
pz = -0.1  # z component of the applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-3

# parameter bounds
qmin = None
qmax = None

# goal horizontal projection
add_horizontal_projection_goal = True

# goal edge length
add_edge_length_goal = False
length_target = 0.03

# goal and constraint edge angle
add_edge_angle_goal = False
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in goal
angle_base = 10.0  # angle constraint, lower bound
angle_top = 30.0  # angle constraint, upper bound

# constraint angle
add_edge_angle_constraint = True
angle_vector_constraint = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_min = 10.0
angle_max = 30.0

# constraint length
add_edge_length_constraint = False
length_min = 0.01
length_max = 0.04

# constraint force
add_edge_force_constraint = False
force_min = -20.0
force_max = 0.0

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

# set initial q to all nodes
for edge in edges_rings:
    network.edge_forcedensity(edge, q0_ring)

for edge in edges_cross:
    network.edge_forcedensity(edge, q0_cross)

# ==========================================================================
# Store network
# ==========================================================================

networks = {"start": network}

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# horizontal projection goal
if add_horizontal_projection_goal:
    for node in network.nodes_free():
        xyz = network.node_coordinates(node)
        line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
        goal = NodeLineGoal(node, target=line)
        goals.append(goal)

# edge length goal
if add_edge_length_goal:
    for cross_ring in edges_cross_rings[:]:
        for edge in cross_ring:
            length = network.edge_length(*edge)
            goal = EdgeLengthGoal(edge, target=length, weight=1.)
            goals.append(goal)

# edge angle goal
if add_edge_angle_goal:
    for i, ring in enumerate(edges_cross_rings):
        angle_delta = angle_top - angle_base
        angle = angle_base + angle_delta * (i / (num_rings - 1))
        print(f"Edges ring {i + 1}/{len(edges_cross_rings)}. Angle goal: {angle}")

        for edge in ring:
            goal = EdgeVectorAngleGoal(edge, vector=angle_vector, target=angle)
            goals.append(goal)

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create constraints
# ==========================================================================

constraints = []
constraint_angles = []

if add_edge_angle_constraint:
    for i, ring in enumerate(edges_cross_rings):
        for edge in ring:
            constraint = EdgeVectorAngleConstraint(edge,
                                                   vector=angle_vector_constraint,
                                                   bound_low=angle_min,
                                                   bound_up=angle_max)
            constraint_angles.append(constraint)
        constraints.extend(constraint_angles)


if add_edge_length_constraint:
    constraints.append(NetworkEdgesLengthConstraint(bound_low=length_min, bound_up=length_max))

if add_edge_force_constraint:
    constraints.append(NetworkEdgesForceConstraint(bound_low=force_min, bound_up=force_max))

# ==========================================================================
# Form-finding sweep
# ==========================================================================

sweep_configs = [{"name": "eq",
                  "method": fdm,
                  "msg": "\n*Form found network*",
                  "save": False},
                 {"name": "eq_g",
                 "method": constrained_fdm,
                  "msg": "\n*Constrained form found network. No constraints*",
                  "save": True},
                 {"name": "eq_g_c",
                  "method": constrained_fdm,
                  "msg": "\n*Constrained form found network. With Constraints*",
                  "save": True,
                  "constraints": constraints}
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
        network = fofin_method(network,
                               optimizer=optimizer(),
                               bounds=(qmin, qmax),
                               loss=loss,
                               constraints=config.get("constraints", []),
                               maxiter=maxiter)

    # store network
    if config["save"]:
        networks[config["name"]] = network

    # Report stats
    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    fields = [q, f, l]
    field_names = ["FDs", "Forces", "Lengths"]

    if constraint_angles:
        model = EquilibriumModel(network)
        q = np.array(network.edges_forcedensities())
        eqstate = model(q)
        a = [constraint.constraint(eqstate, model) for constraint in constraint_angles]
        fields.append(a)
        field_names.append("Angles")

    print(f"Load path: {round(network.loadpath(), 3)}")
    for name, vals in zip(field_names, fields):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# add all networks except the last one
networks = list(networks.values())
for i, network in enumerate(networks):
    if i == (len(networks) - 1):
        continue
    viewer.add(network, show_points=True, linewidth=1.0, color=Color.grey().darkened(i * 10))

network0 = networks[0]
c_network = networks[-1]  # last network is colored

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

    # draw residual forces
    residual = c_network.node_residual(node)

    if length_vector(residual) < 0.001:
        continue

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
# viewer.zoom_extents()
viewer.show()
