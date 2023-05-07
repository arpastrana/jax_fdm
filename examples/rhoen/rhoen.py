# the essentials
import os
from math import ceil, floor
import numpy as np

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.datastructures import Mesh
from compas.datastructures import network_find_cycles

# jax fdm
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import EdgesLengthEqualGoal

from jax_fdm.losses import MeanSquaredError
from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer
from jax_fdm.visualization import Plotter

# ==========================================================================
# Parameters
# ==========================================================================

name = "rhoen"

pole_height = 3.5
q0 = 1.0

px, py, pz = 0.0, 0.0, -0.2  # loads at each node
qmin, qmax = 0.01, 10.0  # min and max force densities

target_length = 0.15
optimizer = LBFGSB  # the optimization algorithm
maxiter = 5000  # optimizer maximum iterations
tol = 1e-9  # optimizer tolerance

optimize = True
record = False  # True to record optimization history of force densities
export = True  # export result to JSON


def round_up(n, decimals):
    multiplier = 10 ** decimals
    return floor(n * multiplier) / multiplier

# ==========================================================================
# Import network
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}_network0.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Define structural system
# ==========================================================================

# data
network.edges_forcedensities(q=q0)
edges_length = [network.edge_length(u, v) for u, v in network.edges()]
print(f"Length: Min: {min(edges_length):.2f} Max: {max(edges_length):.2f} Mean: {(sum(edges_length)/len(edges_length)):.2f} Std: {np.std(np.array(edges_length)):.2f}")

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = network.copy()

# move poles up
for node in network.nodes_where({"is_pole": True}):
    x, y, z = network.node_coordinates(node)
    xyz = [x, y, pole_height]
    network.node_attributes(node, "xyz", xyz)

network = fdm(network)
network_fd = network.copy()

print(f"Load path: {round(network.loadpath(), 3)}")

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_base.json")
    network.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_fd.json")
    network_fd.to_json(FILE_OUT)
    print("FDM-ed network exported to", FILE_OUT)

# ==========================================================================
# Define optimization parameters
# ==========================================================================

if optimize:
    parameters = []
    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

    goals = []

    # target edge lengths
    for edge in network.edges():
        length = network0.edge_length(*edge)
        length = round_up(length, 1)
        # length = target_length
        goal = EdgeLengthGoal(edge, length)
        goals.append(goal)

    # equalize edge lengths
    # goal = EdgesLengthEqualGoal(list(network.edges()), weight=10.0)
    # goals.append(goal)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

    squared_error = SquaredError(goals, alpha=1.0)
    loss = Loss(squared_error)

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

    optimizer = optimizer()
    recorder = OptimizationRecorder(optimizer) if record else None

    network = constrained_fdm(network,
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
# Report stats
# ==========================================================================

    edges_length = [network.edge_length(u, v) for u, v in network.edges()]
    print(f"Length: Min: {min(edges_length):.2f} Max: {max(edges_length):.2f} Mean: {(sum(edges_length)/len(edges_length)):.2f} Std: {np.std(np.array(edges_length)):.2f}")
    network.print_stats()

    # for edge in network.edges():
    #     length = network0.edge_length(*edge)
    #     length0 = round_up(network0.edge_length(*edge), 1)
    #     length1 = network.edge_length(*edge)
    #     print(f"Edge: {edge} Target: {length0:.2f} vs. Optimized: {length1:.2f}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)
# plotter = Plotter(dpi=150)

# modify view
viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.view.camera.rotation[2] = 0.0  # set rotation around z axis to zero

viewer.add(network, edgecolor="fd", show_reactions=False)

# optimized network
# viewer.add(network,
#            as_wireframe=True,
#            show_points=False,
#            linewidth=1.0,
#            color=Color.red())

# plotter.add(network0,
#             edgewidth=0.5,
#             edgecolor="force",
#             show_nodelabel=True,
#             show_loads=False,
#             show_reactions=False,
#             show_nodes=True,
#             nodesize=50.0,
#             loadscale=5.0)

# reference network
# viewer.add(network_fd,
#            as_wireframe=True,
#            show_points=False,
#            linewidth=1.0,
#            color=Color.grey().darkened())

viewer.show()

# show le crème
# plotter.zoom_extents()
# plotter.show()
