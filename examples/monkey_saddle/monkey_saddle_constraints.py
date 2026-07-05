# the essentials
import os

# compas
from compas.colors import Color
from compas.datastructures import Mesh
from compas.itertools import pairwise
from compas.topology import dijkstra_path

# jax fdm
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.optimization import TrustRegionConstrained
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
rmin, rmax = 2.0, 10.0  # min and max reaction forces
r_exp = 0.5  # reaction force variation exponent

add_constraints = True  # input constraints to the optimization problem
length_min = 0.5  # minimum allowed edge length for length constraint
length_max = 4.0  # maximum allowed edge length for length constraint

optimizer = TrustRegionConstrained  # optimization algorithm
maxiter = 500  # optimizer maximum iterations
tol = 1e-2  # optimizer tolerance

record = True  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import coarse mesh
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}.json"))
mesh = FDMesh.from_json(FILE_IN)

print('Initial coarse mesh:', mesh)

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
mesh.vertices_loads([px, py, pz])
mesh.edges_forcedensities(q0)

print("FD mesh:", mesh)

# edges between two anchored vertices are pinned out of the optimization: they
# stay at their initial force density and are not design variables
edges_free = [(u, v) for u, v in mesh.edges() if u not in supports or v not in supports]

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
for edge in edges_free:
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# reaction forces
goals = []
for key in mesh.vertices_supports():
    step = steps[key]
    reaction = (1 - step / max_step) ** r_exp * (rmax - rmin) + rmin
    goal = NodeResidualForceGoal(key, reaction)
    goals.append(goal)

# ==========================================================================
# Define constraints
# ==========================================================================

constraints = None
if add_constraints:
    constraints = []
    for edge in edges_free:
        constraint = EdgeLengthConstraint(edge, bound_low=length_min, bound_up=length_max)
        constraints.append(constraint)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error = SquaredError(goals, name="ReactionForceGoal")
loss = Loss(squared_error)

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

design = constrained_fdm(mesh,
                         optimizer=optimizer,
                         loss=loss,
                         parameters=parameters,
                         constraints=constraints,
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

viewer = Viewer(width=1600, height=900, show_grid=False)

# modify view
viewer.renderer.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.renderer.camera.rotation.z = 0.0  # set rotation around z axis to zero

# view reference mesh as a plain wireframe (convert to a plain COMPAS mesh so the
# viewer renders bare geometry instead of the force-density mesh artist)
viewer.add(mesh.copy(cls=Mesh),
           show_faces=False,
           show_points=False,
           show_edges=True,
           linewidth=1.0,
           color=Color.grey().darkened())

# view the optimized mesh directly: shaded surface plus edges colored by force,
# straight from the FDMesh
viewer.add(design,
           edgewidth=(0.02, 0.2),
           show_nodes=False,
           edgecolor="force",
           show_reactions=False,
           show_loads=False,
           reactionscale=0.75)

# show le crème
viewer.show()
