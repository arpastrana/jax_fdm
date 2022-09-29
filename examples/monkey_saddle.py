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
from compas.topology import dijkstra_path
from compas.utilities import pairwise

# pattern-making
from compas_singular.datastructures import CoarseQuadMesh

# visualization
from compas_view2.app import App

# force density
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.goals import NetworkLoadPathGoal

from jax_fdm.losses import PredictionError
from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss
from jax_fdm.losses import L2Regularizer

# ==========================================================================
# Parameters
# ==========================================================================

name = "monkey_saddle"

n = 4  # densification of coarse mesh

q0 = -2.0
px, py, pz = 0.0, 0.0, -1.0  # loads at each node
qmin, qmax = -20.0, -0.01  # min and max force densities
rmin, rmax = 2.0, 10.0  # min and max reaction forces
r_exp = 0.5  # reaction force variation exponent

weight_length = 1.0  # weight for edge length goal in optimisation
weight_residual = 10.0  # weight for residual force goal in optimisation

alpha = 0.1  # weight of the L2 regularization term in the loss function
alpha_lp = 0.01   # weight of the total load path minimization goal

maxiter = 500  # optimizer maximum iterations
tol = 1e-3  # optimizer tolerance

record = False  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import coarse mesh
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../data/json/{name}.json"))
mesh = CoarseQuadMesh.from_json(FILE_IN)

print('Initial coarse mesh:', mesh)

# ==========================================================================
# Densify coarse mesh
# ==========================================================================

mesh.collect_strips()
mesh.set_strips_density(n)
mesh.densification()
mesh = mesh.get_quad_mesh()
mesh.collect_polyedges()

print("Densified mesh:", mesh)

# ==========================================================================
# Define support conditions
# ==========================================================================

polyedge2length = {}
for pkey, polyedge in mesh.polyedges(data=True):
    if mesh.is_vertex_on_boundary(polyedge[0]) and mesh.is_vertex_on_boundary(polyedge[1]):
        length = sum([mesh.edge_length(u, v) for u, v in pairwise(polyedge)])
        polyedge2length[tuple(polyedge)] = length

supports = []
n = sum(polyedge2length.values()) / len(polyedge2length)
for polyedge, length in polyedge2length.items():
    if length < n:
        supports += polyedge

supports = set(supports)

print("Number of supported nodes:", len(supports))

# ==========================================================================
# Compute assembly sequence (simplified)
# ==========================================================================

steps = {}
corners = set([vkey for vkey in mesh.vertices() if mesh.vertex_degree(vkey) == 2])
adjacency = mesh.adjacency
weight = {(u, v): 1.0 for u in adjacency for v in adjacency[u]}

for vkey in supports:
    if vkey in corners:
        steps[vkey] = 0
    else:
        len_dijkstra = []
        for corner in corners:
            len_dijkstra.append(len(dijkstra_path(adjacency, weight, vkey, corner)) - 1)
        steps[vkey] = min(len_dijkstra)

max_step = max(steps.values())
steps = {vkey: max_step - step for vkey, step in steps.items()}

# ==========================================================================
# Define structural system
# ==========================================================================

nodes = [mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()]
edges = [(u, v) for u, v in mesh.edges() if u not in supports or v not in supports]
network0 = FDNetwork.from_nodes_and_edges(nodes, edges)

print("FD network:", network0)

# data
network0.nodes_supports(supports)
network0.nodes_loads([px, py, pz], keys=network0.nodes())
network0.edges_forcedensities(q=q0)

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_base.json")
    network0.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals_a = []
for edge in network0.edges():
    length = network0.edge_length(*edge)
    goal = EdgeLengthGoal(edge, length, weight=weight_length)
    goals_a.append(goal)

# reaction forces
goals_b = []
for key in network0.nodes_supports():
    step = steps[key]
    reaction = (1 - step / max_step) ** r_exp * (rmax - rmin) + rmin
    goal = NodeResidualForceGoal(key, reaction, weight=weight_residual)
    goals_b.append(goal)

# global loadpath goal
goals_c = []
load_path = NetworkLoadPathGoal()
goals_c.append(load_path)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error_a = SquaredError(goals_a, alpha=1.0, name="EdgeLengthGoal")
squared_error_b = SquaredError(goals_b, alpha=1.0, name="ReactionForceGoal")
loadpath_error = PredictionError(goals_c, alpha=alpha_lp, name="LoadPathGoal")
regularizer = L2Regularizer(alpha=alpha)

loss = Loss(squared_error_a, squared_error_b, loadpath_error, regularizer)

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = fdm(network0)

print(f"Load path: {round(network0.loadpath(), 3)}")

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

recorder = None
if record:
    recorder = OptimizationRecorder()


network = constrained_fdm(network0,
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
viewer.add(network0, show_points=False, linewidth=1.0, color=Color.grey().darkened())

# edges color map
cmap = ColorMap.from_mpl("viridis")

fds = [fabs(network.edge_forcedensity(edge)) for edge in network.edges()]
colors = {}
for edge in network.edges():
    fd = fabs(network.edge_forcedensity(edge))
    ratio = (fd - min(fds)) / (max(fds) - min(fds))
    colors[edge] = cmap(ratio)

# optimized network
viewer.add(network,
           show_vertices=False,
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
