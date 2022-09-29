# the essentials
import os
from math import fabs
import matplotlib.pyplot as plt

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector

# visualization
from compas_view2.app import App

# force density
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss


# ==========================================================================
# Parameters
# ==========================================================================

name = "butt"
name_target = "butt_target"

q0 = -1.0
px, py, pz = 0.0, 0.0, -0.2  # loads at each node
qmin, qmax = -20.0, -0.01  # min and max force densities

maxiter = 1000  # optimizer maximum iterations
tol = 1e-3  # optimizer tolerance

record = False  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import network
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../data/json/{name}.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Import targer network
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(HERE, f"../data/json/{name_target}.json"))
print(FILE_IN)
network_target = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Define structural system
# ==========================================================================

# data
supports = [node for node in network.nodes() if network.is_leaf(node)]
network.nodes_supports(supports)
network.nodes_loads([px, py, pz], keys=network.nodes_free())
network.edges_forcedensities(q=q0)

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_base.json")
    network.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals = []
for node in network.nodes():
    if node in supports:
        continue
    xyz = network_target.node_coordinates(node)
    goal = NodePointGoal(node, xyz)
    goals.append(goal)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error = SquaredError(goals, alpha=1.0)

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

recorder = None
if record:
    recorder = OptimizationRecorder()


network = constrained_fdm(network,
                          optimizer=SLSQP(),
                          loss=loss,
                          bounds=(qmin, qmax),
                          maxiter=maxiter,
                          tol=tol,
                          callback=recorder)

# ==========================================================================
# Export optimization history
# ==========================================================================

if record and export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_history.json")
    recorder.to_json(FILE_OUT)
    print("Optimization history exported to", FILE_OUT)

# ==========================================================================
# Plot loss components
# ==========================================================================

if record:
    model = EquilibriumModel(network)
    fig = plt.figure(dpi=150)
    y = []
    for q in recorder.history:
        error = loss(q, model)
        y.append(error)
    plt.plot(y, label=loss.name)

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
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_optimized.json")
    network.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Report stats
# ==========================================================================

q = list(network.edges_forcedensities())
f = list(network.edges_forces())
l = list(network.edges_lengths())

print(f"Load path: {round(network.loadpath(), 3)}")
for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

    minv = round(min(vals), 3)
    maxv = round(max(vals), 3)
    meanv = round(sum(vals) / len(vals), 3)
    print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# modify view
viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.view.camera.rotation[2] = 0.0  # set rotation around z axis to zero

# reference network
# viewer.add(network_fd, show_points=False, linewidth=1.0, color=Color.black())
viewer.add(network_target, show_points=False, linewidth=1.0, color=Color.grey().darkened())

# edges color map
cmap = ColorMap.from_mpl("viridis")

fds = [fabs(network.edge_forcedensity(edge)) for edge in network.edges()]
colors = {}
for edge in network.edges():
    fd = fabs(network.edge_forcedensity(edge))
    try:
        ratio = (fd - min(fds)) / (max(fds) - min(fds))
    except ZeroDivisionError:
        ratio = 1.
    colors[edge] = cmap(ratio)

# optimized network
viewer.add(network,
           show_vertices=True,
           pointsize=12.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in network.nodes():

    pt = network.node_coordinates(node)

    # draw residual forces
    residual = network.node_residual(node)

    if length_vector(residual) < 0.001:
        continue

    # print(node, residual, length_vector(residual))
    residual_line = Line(pt, add_vectors(pt, residual))
    viewer.add(residual_line,
               linewidth=4.0,
               color=Color.pink())

# draw applied loads
for node in network.nodes():
    pt = network.node_coordinates(node)
    load = network.node_load(node)
    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=4.0,
               color=Color.green().darkened())

# draw supports
for node in network.nodes_supports():
    x, y, z = network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=20)

# show le crème
viewer.show()
