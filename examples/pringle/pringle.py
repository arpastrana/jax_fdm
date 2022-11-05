"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import Translation
from compas.datastructures import network_transform

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodePlaneGoal
from jax_fdm.goals import NodeResidualForceGoal

from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.optimization import EdgeForceDensityParameter

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Initial parameters
# ==========================================================================

name = "pringle"

length_vault = 6.0
width_vault = 3.0

num_u = 10
num_v = 9  # only odd numbers

q_init = -0.25
pz = -0.1

rz_min = 0.45
rz_max = 2.0

record = True
export = True

HERE = os.path.dirname(__file__)

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

# define anchors
for arch in arches:
    network.node_anchor(arch[0])
    network.node_anchor(arch[-1])

# apply loads to unanchored nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all nodes
for edge in network.edges():
    network.edge_forcedensity(edge, q_init)

# center vault around origin
T = Translation.from_vector([-length_vault / 2., -width_vault / 2., 0.])
network_transform(network, T)

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_base.json")
    network.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

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
# Define parameters
# ==========================================================================

parameters = []
for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, -5.0, -0.1)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# residual forces
goals_a = []
for rz, arch in zip(rzs, arches):
    goals_a.append(NodeResidualForceGoal(arch[0], target=rz, weight=100.0))
    goals_a.append(NodeResidualForceGoal(arch[-1], target=rz, weight=100.0))

# transversal planes
goals_b = []
for node in network.nodes_free():
    origin = network.node_coordinates(node)
    normal = [1.0, 0.0, 0.0]
    goal = NodePlaneGoal(node, target=(origin, normal), weight=10.0)
    goals_b.append(goal)

# transversal edge lengths
goals_c = []
for edge in cross_edges:
    target_length = network.edge_length(*edge)
    goals_c.append(EdgeLengthGoal(edge, target=target_length, weight=1.0))

# ==========================================================================
# Create loss function
# ==========================================================================

squared_error_a = SquaredError(goals_a, alpha=1.0, name="ResidualForceGoal")
squared_error_b = SquaredError(goals_b, alpha=1.0, name="NodePlaneGoal")
squared_error_c = SquaredError(goals_c, alpha=1.0, name="EdgeLengthGoal")

loss = Loss(squared_error_a, squared_error_b, squared_error_c)

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

recorder = None
if record:
    recorder = OptimizationRecorder()

c_network = constrained_fdm(network,
                            optimizer=SLSQP(),
                            loss=loss,
                            parameters=parameters,
                            maxiter=200,
                            tol=1e-9,
                            callback=recorder)

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_optimized.json")
    c_network.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

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
# Print out stats
# ==========================================================================

c_network.print_stats()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

# optimized network
viewer.add(c_network,
           edgewidth=(0.02, 0.1),
           loadscale=2.0,
           edgecolor="fd")

# reference network
viewer.add(network,
           as_wireframe=True,
           show_points=False,
           linewidth=2.0,
           color=Color.grey().darkened())

# draw lines betwen subject and target nodes
for node in c_network.nodes():
    pt = c_network.node_coordinates(node)
    target_pt = network.node_coordinates(node)
    viewer.add(Line(target_pt, pt), linewidth=1.0, color=Color.grey().lightened())

# show le crème
viewer.show()
