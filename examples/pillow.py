"""
Solve a constrained force density problem using gradient-based optimization.
"""

import os
import numpy as np
from random import random, choice
from math import fabs
from math import radians
from math import pi, cos, sin, atan

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Vector
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.geometry import discrete_coons_patch
from compas.datastructures import Mesh
from compas.datastructures import mesh_weld
from compas.utilities import pairwise

from compas_singular.datastructures import CoarseQuadMesh
# from compas_quad.datastructures import CoarseQuadMesh

# visualization
from compas_view2.app import App

# static equilibrium
from dfdm.datastructures import FDNetwork

from dfdm.equilibrium import fdm
from dfdm.equilibrium import constrained_fdm
from dfdm.equilibrium import EquilibriumModel

from dfdm.goals import EdgeVectorAngleGoal
from dfdm.goals import EdgeDirectionGoal
from dfdm.goals import EdgeLengthGoal
from dfdm.goals import NodeLineGoal
from dfdm.goals import NodePlaneGoal
from dfdm.goals import NodeResidualForceGoal
from dfdm.goals import NetworkLoadPathGoal

from dfdm.constraints import EdgeVectorAngleConstraint
from dfdm.constraints import NodeNormalAngleConstraint
from dfdm.constraints import NetworkEdgesLengthConstraint
from dfdm.constraints import NetworkEdgesForceConstraint

from dfdm.losses import PredictionError
from dfdm.losses import SquaredError
from dfdm.losses import MeanSquaredError
from dfdm.losses import L2Regularizer
from dfdm.losses import Loss

from dfdm.optimization import SLSQP
from dfdm.optimization import BFGS
from dfdm.optimization import TrustRegionConstrained
from dfdm.optimization import OptimizationRecorder


# ==========================================================================
# Initial parameters
# ==========================================================================

model_name = "pillow"

# geometric parameters
l1, l2 = 10.0, 10.0
divisions = 8

# initial form-finding parameters
q0 = -2.0  # starting force density
pz = -1.0  # z component of the applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-3

# parameter bounds
qmin = None
qmax = None

# goal horizontal projection
add_horizontal_projection_goal = True

# constraint normal angle
add_node_normal_angle_constraint = False
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_min = pi/2.0 - atan(0.75)
angle_max = pi/2.0
print(angle_min, angle_max)

# constraint length
add_edge_length_constraint = False
ratio_length_min = 1.0
ratio_length_max = 3.0

# constraint force
add_edge_force_constraint = False
force_min = -100.0
force_max = 0.0

export = False
view = False

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Create the base geometry of the dome
# ==========================================================================

vertices = [[l1, 0.0, 0.0], [l1, l2, 0.0], [0.0, l2, 0.0], [0.0, 0.0, 0.0]]
faces = [[0, 1, 2, 3]]
coarse = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)
coarse.collect_strips()
coarse.set_strips_density(divisions)
coarse.densification()
mesh = coarse.get_quad_mesh()

vertices, faces = mesh.to_vertices_and_faces()
edges = mesh.edges()
network = FDNetwork.from_nodes_and_edges(vertices, edges)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for key in network.nodes():
    if mesh.is_vertex_on_boundary(key):
        network.node_support(key)

# apply loads
for key in network.nodes():
    network.node_load(key, load=[0.0, 0.0, pz])

# set initial q to all edges
for edge in network.edges():
    network.edge_forcedensity(edge, q0)
    network.edge_forcedensity(edge, q0 + 0.1 * random())

networks = {'input': network}

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

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create constraints
# ==========================================================================

constraints = []
constraint_normals = []

if add_node_normal_angle_constraint:
    for key in network.nodes():
        if not mesh.is_vertex_on_boundary(key):
            polygon = mesh.vertex_neighbors(key, ordered=True)
            constraint = NodeNormalAngleConstraint(key, polygon, angle_vector, bound_low=angle_min, bound_up=angle_max)
            constraints.append(constraint)
            constraint_normals.append(constraint)

if add_edge_length_constraint:
    average_length = np.mean([network.edge_length(*edge) for edge in network.edges()])
    length_min = ratio_length_min * average_length
    length_max = ratio_length_max * average_length
    constraints.append(NetworkEdgesLengthConstraint(bound_low=length_min, bound_up=length_max))

if add_edge_force_constraint:
    constraints.append(NetworkEdgesForceConstraint(bound_low=force_min, bound_up=force_max))

# ==========================================================================
# Form-finding
# ==========================================================================

networks['free'] = fdm(network)

networks['uncstr_opt'] = constrained_fdm(network,
                                         optimizer=optimizer(),
                                         bounds=(qmin, qmax),
                                         loss=loss,
                                         maxiter=maxiter)

networks['cstr_opt'] = constrained_fdm(network,
                                       optimizer=optimizer(),
                                       bounds=(qmin, qmax),
                                       loss=loss,
                                       constraints=constraints,
                                       maxiter=maxiter)

for network_name, network in networks.items():

    if network_name == "input":
        continue

    print("\n Design {}".format(network_name))

    print(f"Load path: {round(network.loadpath(), 3)}")

    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    data = {'Force densities': q, 'Forces': f, 'Lengths': l}

    if constraint_normals:
        model = EquilibriumModel(network)
        q = np.array(network.edges_forcedensities())
        eqstate = model(q)
        a = [constraint.constraint(eqstate, model) for constraint in constraint_normals]
        data['Normal angles'] = a

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
