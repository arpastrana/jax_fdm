# the essentials
import os
from random import random
from random import choice
from math import radians
import numpy as np
from math import fabs
from scipy.spatial.distance import directed_hausdorff

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import dot_vectors
from compas.datastructures import Mesh
from compas.datastructures import network_find_cycles

# jax fdm
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import MeanSquaredError
from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss
from jax_fdm.losses import L2Regularizer

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Parameters
# ==========================================================================

name = "butt"
name_target = "butt_target"

q0 = 0.4  # -2
px, py, pz = 0.0, 0.0, -0.2  # loads at each node
qmin, qmax = -50.0, 0.0  # min and max force densities

alpha_reg = 7e-6  # 7e-6  # 1e-5
error = MeanSquaredError
optimizer = LBFGSB  # the optimization algorithm
maxiter = 10000  # optimizer maximum iterations
tol = 1e-9  # optimizer tolerance

record = False  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import network
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Import target network
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name_target}.json"))
network_target = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Define structural system
# ==========================================================================

# data
supports = [node for node in network.nodes() if network_target.node_attribute(node, "z") <= 0.35]
network.nodes_supports(supports)
# network.nodes_loads([px, py, pz], keys=network.nodes_free())

# single q
network.edges_forcedensities(q=q0)

# random q
# for edge in network.edges():
    # network.edge_forcedensity(edge, q=q0*random()*1.2)# *choice((-1.0, 1.0)))

# q depending on orientation
# for edge in network.edges():
#     factor = fabs(dot_vectors(network.edge_vector(*edge), [0.0, 1.0, 0.0]))
#     if factor <= 0.1:
#         _q = q0 / 10.0
#     else:
#         _q = q0
#     network.edge_forcedensity(edge, q=_q)

# mesh for loads
vertices = {node: network.node_coordinates(node) for node in network.nodes()}
faces = network_find_cycles(network)[1:]
mesh = Mesh.from_vertices_and_faces(vertices, faces)

for node in network.nodes():
    area = mesh.vertex_area(node)
    network.node_load(node, [px * area, py * area, pz * area])

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_base.json")
    network.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Define optimization parameters
# ==========================================================================

parameters = []
for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals = []
for node in network.nodes():
    xyz = network_target.node_coordinates(node)
    goal = NodePointGoal(node, xyz)
    goals.append(goal)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error = error(goals, alpha=1.0)
loss = Loss(squared_error, L2Regularizer(alpha_reg))

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = network.copy()
network = fdm(network, sparse=False)
network_fd = network.copy()

print(f"Load path: {round(network.loadpath(), 3)}")

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

optimizer = optimizer()
recorder = OptimizationRecorder(optimizer) if record else None

# network = constrained_fdm(network0,
#                           optimizer=optimizer,
#                           loss=loss,
#                           parameters=parameters,
#                           maxiter=maxiter,
#                           tol=tol,
#                           callback=recorder)

# ==========================================================================
# Export optimization history
# ==========================================================================

if record and export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_history.json")
    recorder.to_json(FILE_OUT)
    print("Optimization history exported to", FILE_OUT)

# ==========================================================================
# Plot loss components
# ==========================================================================

if record:
    plotter = LossPlotter(loss, network, dpi=150, figsize=(8, 4))
    plotter.plot(recorder.history)
    plotter.show()

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_optimized.json")
    network.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Hausdorff distance
# ==========================================================================

U = np.array([network.node_coordinates(node) for node in network.nodes()])
V = np.array([network_target.node_coordinates(node) for node in network_target.nodes()])
directed_u = directed_hausdorff(U, V)[0]
directed_v = directed_hausdorff(V, U)[0]
hausdorff = max(directed_u, directed_v)

print(f"Hausdorff distances: Directed U: {directed_u}\tDirected V: {directed_v}\tUndirected: {round(hausdorff, 4)}")

# ==========================================================================
# Report stats
# ==========================================================================

network.print_stats()

# ==========================================================================
# Mesh
# ==========================================================================

network = network_fd
vertices = {node: network.node_coordinates(node) for node in network.nodes()}
faces = network_find_cycles(network)[1:]
mesh = Mesh.from_vertices_and_faces(vertices, faces)

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

# modify view
# viewer.view.camera.zoom(-80)  # number of steps, negative to zoom out
# viewer.view.camera.rotation[2] = 0.8  # set rotation around z axis to zero

viewer.view.camera.position = (35.338, -28.023, 32.489)
viewer.view.camera.target = (-0.125, -0.157, -1.006)
viewer.view.camera.distance = 50.0

# initial network
# viewer.add(network0,
#            edgewidth=0.1,
#            edgecolor=Color.grey().darkened(10),
#            show_nodes=True,
#            nodesize=0.3,  # 0.3, 0.4
#            show_reactions=False,
#            show_loads=True,
#            loadscale=2.0)  # 2.0, 1.0

# optimized network
viewer.add(network,
           edgewidth=(0.05, 0.4),
           edgecolor="force",
           reactionscale=0.4,  # 0.15
           show_loads=True,
           loadscale=1.0)

viewer.add(mesh, show_points=False, show_edges=False, opacity=0.5)  # 0.5

# # reference network
# viewer.add(network_target,
#            as_wireframe=True,
#            show_points=False,
#            linewidth=1.0,
#            color=Color.grey().darkened())

# # draw lines to target
# for node in network.nodes():
#     pt = network.node_coordinates(node)
#     line = Line(pt, network_target.node_coordinates(node))
#     viewer.add(line, color=Color.grey())

# show le crème
viewer.show()
