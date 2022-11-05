# the essentials
import os

# compas
from compas.colors import Color
from compas.topology import dijkstra_path
from compas.utilities import pairwise

# pattern-making
from compas_singular.datastructures import CoarseQuadMesh

# force density
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.goals import NetworkLoadPathGoal

from jax_fdm.losses import PredictionError
from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss
from jax_fdm.losses import L2Regularizer

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


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

optimizer = LBFGSB  # optimization algorithm
maxiter = 500  # optimizer maximum iterations
tol = 1e-3  # optimizer tolerance

record = True  # True to record optimization history of force densities
export = True  # export result to JSON

# ==========================================================================
# Import coarse mesh
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}.json"))
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
network0.nodes_loads([px, py, pz], keys=network0.nodes_free())
network0.edges_forcedensities(q=q0)

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_base.json")
    network0.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Define parameters
# ==========================================================================

parameters = []
for edge in network0.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

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
                          optimizer=optimizer(),
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
    plotter = LossPlotter(loss, network, dpi=150)
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
           edgewidth=(0.05, 0.25),
           reactionscale=0.75,
           edgecolor="fd")

# reference network
viewer.add(network0,
           as_wireframe=True,
           show_points=False,
           linewidth=1.0,
           color=Color.grey().darkened())

# show le crème
viewer.show()
