"""
Solve a constrained force density problem using gradient-based optimization.
"""
from math import fabs

import matplotlib.pyplot as plt

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import Translation
from compas.datastructures import network_transform

# visualization
from compas_view2.app import App

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import EquilibriumModel

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodePlaneGoal
from jax_fdm.goals import NodeResidualForceGoal

from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import OptimizationRecorder


# ==========================================================================
# Initial parameters
# ==========================================================================

length_vault = 6.0
width_vault = 3.0

num_u = 10
num_v = 9  # only odd numbers

q_init = -0.25
pz = -0.1

rz_min = 0.45
rz_max = 2.0

record = False

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Create the base geometry of the vault
# ==========================================================================

xyz_origin = [0.0, 0.0, 0.0]
length_u = length_vault / (num_u - 1)
length_v = width_vault / (num_v - 1)

arches = []
long_edges = []
for i in range(num_v):

    arch = []
    start = add_vectors(xyz_origin, [0.0, i * length_v, 0.0, 0.0])

    for j in range(num_u):

        x, y, z = add_vectors(start, [j * length_u, 0.0, 0.0])
        node = network.add_node(x=x, y=y, z=z)
        arch.append(node)

    arches.append(arch)

    a = i * num_u
    b = a + num_u - 1
    for u, v in zip(range(a, b), range(a + 1, b + 1)):
        edge = network.add_edge(u, v)
        long_edges.append(edge)

cross_edges = []
for i in range(1, num_u - 1):
    seq = []
    for arch in arches:
        seq.append(arch[i])
    for u, v in zip(seq[:-1], seq[1:]):
        edge = network.add_edge(u, v)
        cross_edges.append(edge)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for arch in arches:
    network.node_support(arch[0])
    network.node_support(arch[-1])

# apply loads to unsupported nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all nodes
for edge in network.edges():
    network.edge_forcedensity(edge, q_init)

# center vault around origin
T = Translation.from_vector([-length_vault / 2., -width_vault / 2., 0.])
network_transform(network, T)

# ==========================================================================
# Create a target distribution of residual force magnitudes
# ==========================================================================

assert num_v % 2 != 0
num_steps = (num_v - 1) / 2.0
step_size = (rz_max - rz_min) / num_steps

rzs = []
for i in range(int(num_steps) + 1):
    rzs.append(rz_min + i * step_size)

rzs = rzs + rzs[0:-1][::-1]

# ==========================================================================
# Define goals
# ==========================================================================

goals = []

# residual forces
for rz, arch in zip(rzs, arches):
    goals.append(NodeResidualForceGoal(arch[0], target=rz, weight=100.0))
    goals.append(NodeResidualForceGoal(arch[-1], target=rz, weight=100.0))

# transversal planes
for node in network.nodes_free():
    origin = network.node_coordinates(node)
    normal = [1.0, 0.0, 0.0]
    goal = NodePlaneGoal(node, target=(origin, normal), weight=10.0)
    goals.append(goal)

# transversal edge lengths
for edge in cross_edges:
    target_length = network.edge_length(*edge)
    goals.append(EdgeLengthGoal(edge, target=target_length, weight=1.0))

# ==========================================================================
# Create loss function
# ==========================================================================

loss = Loss(SquaredError(goals))

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

recorder = None
if record:
    recorder = OptimizationRecorder()

c_network = constrained_fdm(network,
                            optimizer=SLSQP(),
                            loss=loss,
                            bounds=(-5.0, -0.1),
                            maxiter=200,
                            tol=1e-9,
                            callback=recorder)

# ==========================================================================
# Plot loss components
# ==========================================================================

if record:
    model = EquilibriumModel(network)
    fig = plt.figure(dpi=150)
    y = []
    for q in recorder.history:
        eqstate = model(q)
        error = loss(eqstate, model)
        y.append(error)
    plt.plot(y, label=loss.__class__.__name__)

    plt.xlabel("Optimization iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

# ==========================================================================
# Print out stats
# ==========================================================================

q = [c_network.edge_forcedensity(edge) for edge in c_network.edges()]
f = [c_network.edge_force(edge) for edge in c_network.edges()]
l = [c_network.edge_length(*edge) for edge in c_network.edges()]

for name, vals in zip(("Q", "Forces", "Lengths"), (q, f, l)):

    minv = round(min(vals), 3)
    maxv = round(max(vals), 3)
    meanv = round(sum(vals) / len(vals), 3)
    print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# reference network
viewer.add(network, show_points=True, linewidth=2.0, color=Color.grey().darkened())

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
           pointsize=12.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in c_network.nodes():

    pt = c_network.node_coordinates(node)

    # draw lines betwen subject and target nodes
    target_pt = network.node_coordinates(node)
    viewer.add(Line(target_pt, pt), linewidth=1.0, color=Color.grey().lightened())

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
    viewer.add(Point(x, y, z), color=Color.green(), size=20)

# show le crème
viewer.show()
