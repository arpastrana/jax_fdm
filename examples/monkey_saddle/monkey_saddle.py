# the essentials
import os

# compas
from compas.datastructures import Mesh
from compas.itertools import pairwise
from compas.topology import dijkstra_path

# jax fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import MeshLoadPathGoal
from jax_fdm.goals import VertexResidualForceGoal
from jax_fdm.losses import L2Regularizer
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer

# ==========================================================================
# Parameters
# ==========================================================================

name = "monkey_saddle"

k = 2  # subdivision levels for coarse mesh densification

q0 = -2.0  # initial force density
px, py, pz = 0.0, 0.0, -1.0  # loads at each node
qmin, qmax = -20.0, -0.01  # min and max force densities
rmin, rmax = 4.0, 8.0  # min and max reaction forces
r_exp = 0.5  # reaction force variation exponent

weight_length = 1.0  # weight for edge length goal in optimisation
weight_residual = 10.0  # weight for residual force goal in optimisation

alpha = 0.1  # weight of the L2 regularization term in the loss function
alpha_lp = 0.01  # weight of the total load path minimization goal

optimizer = LBFGSB  # optimization algorithm
maxiter = 500  # optimizer maximum iterations
tol = 1e-3  # optimizer tolerance

record = True  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import coarse mesh
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}.json"))
mesh = FDMesh.from_json(FILE_IN)

print("Initial coarse mesh:", mesh)

# ==========================================================================
# Densify coarse mesh
# ==========================================================================

mesh = mesh.subdivided(scheme="quad", k=k)

print("Densified mesh:", mesh)

# ==========================================================================
# Define anchor conditions
# ==========================================================================

# corners are the degree-2 vertices; they split the boundary into polyedges
corners = set([vkey for vkey in mesh.vertices() if mesh.vertex_degree(vkey) == 2])

# walk the boundary loop and split it into polyedges at the corners
boundary = mesh.vertices_on_boundaries()[0]
if boundary[0] == boundary[-1]:
    boundary = boundary[:-1]
start = next(i for i, vkey in enumerate(boundary) if vkey in corners)
boundary = boundary[start:] + boundary[:start]

polyedges = []
polyedge = [boundary[0]]
for vkey in boundary[1:] + [boundary[0]]:
    polyedge.append(vkey)
    if vkey in corners:
        polyedges.append(polyedge)
        polyedge = [vkey]

polyedge2length = {}
for polyedge in polyedges:
    length = sum([mesh.edge_length((u, v)) for u, v in pairwise(polyedge)])
    polyedge2length[tuple(polyedge)] = length

# anchor the polyedges that are shorter than the mean boundary polyedge
supports = []
mean_length = sum(polyedge2length.values()) / len(polyedge2length)
for polyedge, length in polyedge2length.items():
    if length < mean_length:
        supports += polyedge

supports = set(supports)

print("Number of support nodes:", len(supports))

# ==========================================================================
# Compute assembly sequence (simplified)
# ==========================================================================

steps = {}
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

# the densified mesh is already an FDMesh; equip it with supports, loads and
# initial force densities
for vkey in supports:
    mesh.vertex_support(vkey)
mesh.vertices_loads([px, py, pz], keys=list(mesh.vertices_free()))
mesh.edges_forcedensities(q0)

print("FD mesh:", mesh)

# ==========================================================================
# Export FD mesh with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_base.json")
    mesh.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Define parameters
# ==========================================================================

parameters = []
for edge in mesh.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals_a = []
for edge in mesh.edges():
    length = mesh.edge_length(edge)
    goal = EdgeLengthGoal(edge, length, weight=weight_length)
    goals_a.append(goal)

# reaction forces
goals_b = []
for key in mesh.vertices_supports():
    step = steps[key]
    reaction = (1 - step / max_step) ** r_exp * (rmax - rmin) + rmin
    goal = VertexResidualForceGoal(key, reaction, weight=weight_residual)
    goals_b.append(goal)

# global loadpath goal
goals_c = []
load_path = MeshLoadPathGoal()
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
# Form-find mesh
# ==========================================================================

mesh = fdm(mesh)

print(f"Load path: {round(mesh.loadpath(), 3)}")

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

optimizer = optimizer()
recorder = OptimizationRecorder(optimizer) if record else None

design = constrained_fdm(
    mesh,
    optimizer=optimizer,
    loss=loss,
    parameters=parameters,
    maxiter=maxiter,
    tol=tol,
    callback=recorder,
)

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
    plotter = LossPlotter(loss, design, dpi=150)
    plotter.plot(recorder.history)
    plotter.show()

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_optimized.json")
    design.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Report stats
# ==========================================================================

design.print_stats()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer()

# modify view
viewer.renderer.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.renderer.camera.rotation.z = 0.0  # set rotation around z axis to zero

# view reference mesh as a plain wireframe (convert to a plain COMPAS mesh)
viewer.add(mesh.copy(cls=Mesh), show_faces=False, show_points=False, show_edges=True)

# view the optimized mesh directly
viewer.add(
    design,
    edgewidth=(0.02, 0.2),
    show_vertices=True,
    show_loads=True,
    show_faces=True,
    edgecolor="fd",
    show_reactions=True,
    reactionscale=0.75,
)

# show le crème
viewer.show()
