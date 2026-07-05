"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os
from random import random

import numpy as np

# compas
from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import add_vectors
from jax_fdm.constraints import EdgeForceConstraint
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.constraints import NodeCurvatureConstraint

# jax_fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.goals import NodeLineGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import SLSQP
from jax_fdm.visualization import Viewer

# ==========================================================================
# Initial parameters
# ==========================================================================

model_name = "pillow"

# geometric parameters
l1 = 10.0
l2 = 10.0
divisions = 8

# initial form-finding parameters
q0, dq = -2.0, 0.1  # starting average force density and random deviation
pz = -100.0  # z component of the total applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-3

# goal horizontal projection
add_horizontal_projection_goal = True
weight_horizontal_projection = 1.0

# goal load path
add_load_path_goal = False
normalise_by_edge_number = False
weight_load_path = 0.001

# goal edge length
add_edge_length_goal = False
weight_edge_length = 1.0

# constraint length
add_edge_length_constraint = True
ratio_length_min = 0.5
ratio_length_max = 3.0

# constraint force
add_edge_force_constraint = True
force_min = -100.0
force_max = -1.0

# constraint curvature
add_curvature_constraint = True
crv_min = -100.0
crv_max = -0.1

export = False

# ==========================================================================
# Create base geometry
# ==========================================================================

mesh = FDMesh.from_meshgrid(dx=l1, nx=divisions, dy=l2, ny=divisions)

# ==========================================================================
# Define structural system
# ==========================================================================

# anchor the boundary vertices
for key in mesh.vertices():
    if mesh.is_vertex_on_boundary(key):
        mesh.vertex_support(key)

# set initial q to all edges
for edge in mesh.edges():
    q = q0 + dq * (random() - 0.5)
    mesh.edge_forcedensity(edge, q)

meshes = {"input": mesh}

# ==========================================================================
# Initial form finding - no external loads
# ==========================================================================

meshes["unloaded"] = fdm(mesh)

# ==========================================================================
# Initial form finding - loaded
# ==========================================================================

mesh_area = mesh.area()
for key in mesh.vertices():
    mesh.vertex_load(key, load=[0.0, 0.0, pz * mesh.vertex_area(key) / mesh_area])

meshes["loaded"] = fdm(mesh)

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# horizontal projection goal
if add_horizontal_projection_goal:
    print("Horizontal projection goal")
    for vertex in mesh.vertices_free():
        xyz = mesh.vertex_coordinates(vertex)
        line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
        goal = NodeLineGoal(vertex, target=line, weight=weight_horizontal_projection)
        goals.append(goal)

# load path goal
if add_load_path_goal:
    if normalise_by_edge_number:
        weight_load_path /= mesh.number_of_edges()
    goals.append(NetworkLoadPathGoal(target=0.0, weight=weight_load_path))

# edge length goal
if add_edge_length_goal:
    mesh_loaded = meshes["loaded"]
    for edge in mesh.edges():
        goal = EdgeLengthGoal(edge, mesh_loaded.edge_length(edge), weight=weight_edge_length)
        goals.append(goal)

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create hard constraints
# ==========================================================================

constraints = []

if add_edge_length_constraint:
    average_length = np.mean([mesh.edge_length(edge) for edge in mesh.edges()])
    length_min = ratio_length_min * average_length
    length_max = ratio_length_max * average_length

    for edge in mesh.edges():
        constraint = EdgeLengthConstraint(edge,
                                          bound_low=length_min,
                                          bound_up=length_max)
        constraints.append(constraint)

    msg = "Edge length constraint between {} and {}"
    print(msg.format(round(length_min, 2), round(length_max, 2)))

if add_edge_force_constraint:
    for edge in mesh.edges():
        constraint = EdgeForceConstraint(edge,
                                         bound_low=force_min,
                                         bound_up=force_max)
        constraints.append(constraint)

    msg = "Edge force constraint between {} and {}"
    print(msg.format(round(force_min, 2), round(force_max, 2)))

if add_curvature_constraint:
    stride = divisions + 1
    mid_column = divisions // 2
    subpolyedge = [mid_column * stride + row for row in range(1, divisions)]

    for key in subpolyedge:
        polygon = mesh.vertex_neighbors(key, ordered=True)
        constraint = NodeCurvatureConstraint(key,
                                             polygon,
                                             bound_low=crv_min,
                                             bound_up=crv_max)
        constraints.append(constraint)

    msg = "Node curvature constraint between {} and {} on {} nodes"
    print(msg.format(round(crv_min, 2), round(crv_max, 2), len(subpolyedge)))

# ==========================================================================
# Form finding
# ==========================================================================

meshes["free"] = fdm(mesh)

meshes["uncstr_opt"] = constrained_fdm(mesh,
                                       optimizer=optimizer(),
                                       loss=loss,
                                       maxiter=maxiter)

meshes["cstr_opt"] = constrained_fdm(mesh,
                                     optimizer=optimizer(),
                                     loss=loss,
                                     constraints=constraints,
                                     maxiter=maxiter)

# ==========================================================================
# Print and export results
# ==========================================================================

for mesh_name, design in meshes.items():
    print()
    print(f"Design {mesh_name}")
    design.print_stats()

    if export:
        HERE = os.path.dirname(__file__)
        FILE_OUT = os.path.join(HERE, f"../data/json/{model_name}_{mesh_name}.json")
        design.to_json(FILE_OUT)
        print(f"Design {mesh_name} exported to {FILE_OUT}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

designs = list(meshes.values())

mesh0 = designs[0]  # reference (input) mesh
c_mesh = designs[-1] if len(designs) > 1 else designs[0]  # optimized design

# view reference mesh as a plain wireframe (convert to a plain COMPAS mesh so the
# viewer renders bare geometry instead of the force-density mesh artist)
viewer.add(mesh0.copy(cls=Mesh),
           show_faces=False,
           show_points=False,
           show_edges=True,
           linewidth=1.0,
           color=Color.grey().darkened(10))

# view the optimized mesh directly: shaded surface plus edges colored by force,
# straight from the FDMesh
viewer.add(c_mesh,
           edgewidth=(0.05, 0.2),
           show_nodes=False,
           edgecolor="force",
           show_reactions=False,
           show_loads=False,
           loadscale=0.5,
           reactionscale=0.5)

# show le crème
viewer.show()
