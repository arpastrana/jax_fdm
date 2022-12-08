"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os

from math import radians

from itertools import cycle

import jax.numpy as jnp
from jax_fdm.geometry import angle_vectors

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import Translation
from compas.geometry import offset_line
from compas.geometry import dot_vectors
from compas.datastructures import network_transform

# compas
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.geometry import Translation
from compas.utilities import pairwise

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.goals import NodePointGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.goals import NodeYCoordinateGoal
from jax_fdm.goals import NodePlaneGoal
from jax_fdm.goals import EdgeDirectionGoal
from jax_fdm.goals import EdgeAngleGoal

from jax_fdm.constraints import NodeZCoordinateConstraint

from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.losses import L2Regularizer

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import TrustRegionConstrained

from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import NodeAnchorXParameter
from jax_fdm.parameters import NodeAnchorYParameter

from jax_fdm.constraints import EdgeLengthConstraint

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Helper functions
# ==========================================================================

def add_arch(network, line, num_segments):
    start, end = line

    nodes= []
    points = Polyline([start, end]).divide(num_segments)
    for x, y, z in points:
        node = network.add_node(x=x, y=y, z=z)
        nodes.append(node)

    edges = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        edge = network.add_edge(u, v)
        edges.append(edge)

    return nodes, edges


def connect_arches(network, arches):
    assert len(arches) == 2
    archa, archb = arches
    assert len(archa) == len(archb)

    edges = []
    for u, v in zip(archa, archb):
        edge = network.add_edge(u, v)
        edges.append(edge)
    return edges


# ==========================================================================
# Initial parameters
# ==========================================================================

name = "pringle"

length_vault = 6.0
course_width = 0.2

num_segments = 11
num_courses = 4

height_arch0 = 1.0

q0_arch = -1
q0_cross = -0.1
pz = -0.1

qmin = None  # -50.0 # -5
qmax = 0.0  # -0.5
xtol = 0.2
ytol = xtol

optimizer = LBFGSB  # LBFGSB
maxiter = 10000
tol = 1e-14

optimize_twice = False
optimizer_2 = TrustRegionConstrained
maxiter_2 = 1000
tol_2 = 1e-9

# goal length
target_length = course_width

# goal vector, angle
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute target angle
angle_base = 40.0
angle_top = 20.0
angle_linear_range = True

record = False
export = False

HERE = os.path.dirname(__file__)

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Initial geometry
# ==========================================================================

xyz_origin = [0.0, 0.0, 0.0]

# create skeleton line
line = Line(xyz_origin, add_vectors(xyz_origin, [0.0, length_vault, 0.0]))
arch_nodes, arch_edges = add_arch(network, line, num_segments)

# assign supports
network.node_anchor(arch_nodes[0])
network.node_anchor(arch_nodes[-1])

# assign loads
for node in network.nodes_free():
    network.node_load(node, [0.0, 0.0, pz])

# assign force densities
for edge in arch_edges:
    network.edge_forcedensity(edge, q0_arch)

# form find initial skeleton
goals = []
constraints = []
for node in network.nodes_free():
    x, y, z = network.node_coordinates(node)
    goal = NodeYCoordinateGoal(node, y)
    goals.append(goal)
    constraint = NodeZCoordinateConstraint(node, 0.0, height_arch0)
    constraints.append(constraint)

loss = Loss(SquaredError(goals))

print("Form-finding central spine...")
network = constrained_fdm(network,
                          loss=loss,
                          constraints=constraints,
                          optimizer=SLSQP())

print(f"Starting arch height: {max(network.nodes_attribute(name='z'))}")

networks = {-1: network.copy()}

# ==========================================================================
# Define structural system
# ==========================================================================

lines = [line] * 2
arches = [arch_nodes] * 2

angles_iter = cycle((angle_base, angle_top))
angle_delta = angle_top - angle_base

for i in range(num_courses):

    arches_new = []
    lines_new = []

    arch_edges_set = []
    cross_edges_set = []
    arch_nodes_set = []

    for arch_nodes, line, sign in zip(arches, lines, (-1, 1)):

        line = offset_line(line, sign * course_width)
        arch_nodes_new, arch_edges = add_arch(network, line, num_segments)
        cross_edges = connect_arches(network, (arch_nodes, arch_nodes_new))
        arch_nodes = arch_nodes_new

# ==========================================================================
# Define structural system
# ==========================================================================

        for edge in arch_edges:
            network.edge_forcedensity(edge, q0_arch)

        for edge in cross_edges:
            network.edge_forcedensity(edge, q0_cross)

        for node in arch_nodes[1:-1]:
            network.node_load(node, load=[0.0, 0.0, pz])

        # assign supports
        network.node_anchor(arch_nodes[0])
        network.node_anchor(arch_nodes[-1])

# ==========================================================================
# Define structural system
# ==========================================================================

        arches_new.append(arch_nodes)
        lines_new.append(line)

        arch_nodes_set.extend(arch_nodes)
        arch_edges_set.extend(arch_edges)
        cross_edges_set.extend(cross_edges)

    # set new arches as old arches
    arches = arches_new
    lines = lines_new

# ==========================================================================
# Define parameters
# ==========================================================================

    parameters = []
    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

    # for node in network.nodes_fixed():
    #     x, y, z = network.node_coordinates(node)
    #     parameters.append(NodeAnchorXParameter(node, x - xtol, x + xtol))
    #     parameters.append(NodeAnchorYParameter(node, y - ytol, y + ytol))

# ==========================================================================
# Create goals
# ==========================================================================

    anchors = list(network.nodes_anchors())

    # node xyz goal
    goals_point = []
    for node in network.nodes_free():
        if node in arch_nodes_set:
            continue
        point = network.node_coordinates(node)
        goal = NodePointGoal(node, point, weight=1.)
        goals_point.append(goal)

    # edge length goal
    goals_length = []
    for edge in cross_edges_set:
        u, v = edge
        if u in anchors or v in anchors:
            continue
        goal = EdgeLengthGoal(edge, target=target_length, weight=1.)
        goals_length.append(goal)

    # transversal planes
    # goals_plane = []
    # for node in arch_nodes_set:
    #     origin = network.node_coordinates(node)
    #     normal = [1.0, 0.0, 0.0]
    #     goal = NodePlaneGoal(node, target=(origin, normal), weight=1.0)
    #     goals_plane.append(goal)

    # edge direction goal

    if angle_linear_range:
        if num_courses == 1:
            num_courses = 2
        angle = angle_base + angle_delta * ((i) / (num_courses - 1))
    else:
        angle = next(angles_iter)
    print(f"\nCourse {i + 1} /{num_courses}. Angle goal: {angle:.2f}")

    goals_vector = []
    goals_angle = []
    vectors_edges = []
    for u, v in cross_edges_set:

        if u in anchors and v in anchors:
            continue

        goal = EdgeAngleGoal((u, v), vector=[0.0, 0.0, 1.0], target=radians(angle))
        goals_angle.append(goal)

        xu, yu, _ = network.node_coordinates(u)  # xyz of first node, assumes it is the lowermost
        u_xyz = [xu, yu, 0.0]

        xv, yv, _ = network.node_coordinates(v)  # xyz of first node, assumes it is the lowermost
        v_xyz = [xv, yv, 0.0]

        angle_rot = angle
        vec = subtract_vectors(v_xyz, u_xyz)
        if dot_vectors(vec, [1., 0., .0]) < 0.0:
            angle_rot = -angle

        vecref = jnp.array([0.0, 0.0, 1.0])

        # print(f"Edge: {u, v}\tAngle:{angle_vectors(jnp.array(vec), vecref)}")

        point = [0.0, 0.0, 1.0]
        end = rotate_points([point], radians(angle_rot), axis=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0]).pop()
        vector = subtract_vectors(end, [0.0, 0.0, 0.0])

        edge = (u, v)
        goal = EdgeDirectionGoal(edge, target=vector, weight=1.)
        goals_vector.append(goal)
        vectors_edges.append((vector, edge))

# ==========================================================================
# Define loss function with goals
# ==========================================================================

    alphal2 = 0.1
    # alphal2 = (0.1 / (i + 1)) ** 2
    if i == num_courses - 1:
        alphal2 = 0.0  # alphal2 / 100.0
    print(alphal2)
    loss = Loss(
                SquaredError(goals=goals_point, name="PointGoal"),
                SquaredError(goals=goals_angle, name="EdgeAngleGoal", alpha=10.),
                # SquaredError(goals=goals_vector, name="EdgeDirectionGoal"),
                SquaredError(goals=goals_length, name="LengthGoal", alpha=1.),
                L2Regularizer(name="L2QRegularizer", alpha=alphal2)
    )


# ==========================================================================
# Constraints
# ==========================================================================

    constraints = []

    # for u, v in cross_edges_set:

    #     if u in anchors and v in anchors:
    #         continue

    #     edge = (u, v)
    #     constraint = EdgeLengthConstraint(edge, course_width * 0.7, course_width * 1.3)
    #     constraints.append(constraint)

# ==========================================================================
# Constrained form-finding
# ==========================================================================

    opt = optimizer()

    recorder = OptimizationRecorder(opt) if record else None

    fnetwork = fdm(network)
    network = constrained_fdm(network,
                              optimizer=opt,
                              parameters=parameters,
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol,
                              constraints=constraints,
                              callback=recorder)

    if optimize_twice:
        network = constrained_fdm(network,
                                  optimizer=optimizer_2(),
                                  parameters=parameters,
                                  loss=loss,
                                  maxiter=maxiter_2,
                                  tol=tol_2,
                                  callback=recorder)
    # Report stats
    network.print_stats()

    lengths = []
    for edge in cross_edges_set:
        lengths.append(network.edge_length(*edge))
    print(f"Average edge length: {sum(lengths) / len(lengths) }")

    # store network
    networks[i] = network.copy()

# ==========================================================================
# Plot loss components
# ==========================================================================

    if record:
        print("\n")
        plotter = LossPlotter(loss, network, dpi=150, figsize=(8, 4))
        plotter.plot(recorder.history)
        plotter.show()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=True)


for i, network in enumerate(networks.values()):
    T = Translation.from_vector([2.0 * i, 0.0, 0.0])
    network.transform(T)
    viewer.add(network,
               edgewidth=(0.01, 0.05),
               loadscale=2.0,
               edgecolor="force")


    # mesh = Mesh.from_lines([network.edge_coordinates(*edge) for edge in network.edges()],
    #                         delete_boundary_face=True)
    # viewer.add(mesh, show_points=False, show_lines=False, opacity=0.5)



# viewer.add(fnetwork, as_wireframe=True)

# add target vectors
# network = networks[num_courses-1]
# for vector, edge in vectors_edges:
#     u, v = edge
#     xyz = network.node_coordinates(u)
#     viewer.add(Line(xyz, add_vectors(xyz, scale_vector(vector, 0.5))))
viewer.show()

# # reference network
# viewer.add(network,
#            as_wireframe=True,
#            show_points=False,
#            linewidth=2.0,
#            color=Color.grey().darkened())

# from compas_view2.shapes import Arrow
# from compas.colors import Color
# from compas.geometry import scale_vector, angle_vectors
# from jax_fdm.geometry import normal_polygon
# # from jax_fdm.geometry import angle_vectors as jax_angle_vectors
# import jax.numpy as jnp


# angles_mesh = []
# for vkey in mesh.vertices():
#     xyz = mesh.vertex_coordinates(vkey)
#     if xyz[2] < 0.1:
#         continue
#     normal = mesh.vertex_normal(vkey)
#     normal = scale_vector(normal, 0.25)
#     angle = angle_vectors([0.0, 0.0, 1.0], normal, deg=True)
#     angles_mesh.append(angle)
#     arrow = Arrow(xyz, normal)
#     viewer.add(arrow, facecolor=Color.black(), show_edges=False, opacity=0.5)

# angles_network = []
# for vkey in mesh.vertices():
#     xyz = mesh.vertex_coordinates(vkey)
#     if xyz[2] < 0.1:
#         continue
#     nbrs = mesh.vertex_neighbors(vkey, ordered=True)
#     polygon = jnp.array([mesh.vertex_coordinates(n) for n in nbrs])
#     normal = normal_polygon(polygon)
#     normal = scale_vector(normal, -0.25)  # NOTE: had to revert normal!
#     angle = angle_vectors(jnp.array([0., 0., 1.]), normal, deg=True)
#     angles_network.append(angle)
#     arrow = Arrow(xyz, normal)
#     viewer.add(arrow, facecolor=Color.red(), show_edges=False, opacity=0.5)

# for ana, anb in zip(angles_mesh, angles_network):
#     print(f"Angle mesh: {ana:.2f}\tAngle network: {anb:.2f}\tDifference: {ana - anb:.2f}")
# # draw lines betwen subject and target nodes
# # for node in c_network.nodes():
# #     pt = c_network.node_coordinates(node)
# #     target_pt = network.node_coordinates(node)
# #     viewer.add(Line(target_pt, pt), linewidth=1.0, color=Color.grey().lightened())

# # show le crème
# viewer.show()
