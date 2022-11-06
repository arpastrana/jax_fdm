# the essentials
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# compas
from compas.colors import Color
from compas.geometry import Line

# jax fdm
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import RootMeanSquaredError
from jax_fdm.losses import Loss

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Parameters
# ==========================================================================

name = "butt"
name_target = "butt_target"

q0 = -2.0
px, py, pz = 0.0, 0.0, -0.2  # loads at each node
qmin, qmax = -20.0, -0.0  # min and max force densities

optimizer = LBFGSB  # the optimization algorithm
maxiter = 1000  # optimizer maximum iterations
tol = 1e-6  # optimizer tolerance

record = True  # True to record optimization history of force densities
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
anchors = [node for node in network.nodes() if network.is_leaf(node)]
network.nodes_anchors(anchors)
network.nodes_loads([px, py, pz], keys=network.nodes_free())
network.edges_forcedensities(q=q0)

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
    if node in anchors:
        continue
    xyz = network_target.node_coordinates(node)
    goal = NodePointGoal(node, xyz)
    goals.append(goal)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error = RootMeanSquaredError(goals, alpha=1.0)
loss = Loss(squared_error)

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = network.copy()
network = fdm(network)
network_fd = network.copy()

print(f"Load path: {round(network.loadpath(), 3)}")

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

optimizer = optimizer()
recorder = OptimizationRecorder(optimizer) if record else None

network = constrained_fdm(network0,
                          optimizer=optimizer,
                          loss=loss,
                          parameters=parameters,
                          maxiter=maxiter,
                          tol=tol,
                          callback=recorder)

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
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

# modify view
viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.view.camera.rotation[2] = 0.0  # set rotation around z axis to zero

# optimized network
viewer.add(network,
           edgewidth=(0.1, 0.3),
           edgecolor="fd",
           loadscale=5.0)

# reference network
viewer.add(network_target,
           as_wireframe=True,
           show_points=False,
           linewidth=1.0,
           color=Color.grey().darkened())

# draw lines to target
for node in network.nodes():
    pt = network.node_coordinates(node)
    line = Line(pt, network_target.node_coordinates(node))
    viewer.add(line, color=Color.grey())

# show le crème
viewer.show()
