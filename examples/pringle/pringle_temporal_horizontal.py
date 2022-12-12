"""
Solve a constrained force density problem using gradient-based optimization.
"""
import os

from math import radians

from itertools import cycle

import jax.numpy as jnp
from jax_fdm.geometry import angle_vectors
from compas_view2.shapes import Arrow
from compas.colors import Color, ColorMap
from compas.geometry import scale_vector, angle_vectors
from jax_fdm.geometry import normal_polygon

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import Translation
from compas.geometry import offset_line
from compas.geometry import dot_vectors
from compas.datastructures import network_transform, network_find_cycles

# compas
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import subtract_vectors, normalize_vector
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.geometry import offset_polyline
from compas.geometry import Translation
from compas.geometry import project_point_plane
from compas.utilities import pairwise
from compas.utilities import geometric_key

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
from jax_fdm.goals import NodeXCoordinateGoal
from jax_fdm.goals import NodeYCoordinateGoal
from jax_fdm.goals import NodeNormalAngleGoal
from jax_fdm.goals import NodeTangentAngleGoal

from jax_fdm.constraints import NodeZCoordinateConstraint
from jax_fdm.constraints import NodeTangentAngleConstraint

from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.losses import L2Regularizer

from jax_fdm.optimization import SLSQP
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import IPOPT
from jax_fdm.optimization import TrustRegionConstrained

from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import NodeAnchorXParameter
from jax_fdm.parameters import NodeAnchorYParameter

from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.constraints import EdgeAngleConstraint

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

num_segments = 10
num_courses = 5

height_arch0 = 1.0

q0_arch = -1.0
q0_cross = -0.1
pz_3d = -0.1

# constrained form-finding, unconstrained optimization
qmin = None
qmax = 0.0

include_supports_as_params = True
xtol = 0.5
ytol = xtol

optimize = True
optimizer = LBFGSB
maxiter = 10000
tol = 1e-14

# goal length
target_length_2d = course_width * 0.5  # for middle cross edges in 2D projection
target_length_3d = course_width # for middle cross edges in 3D

# goal tangent angle and edge direction
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute target angle
angle_base = 20.0
angle_top = 40.0
angle_linear_range = True

# constrained form-finding, unconstrained optimization
add_constraints = False
length_delta = 0.25
optimizer_2 = SLSQP
maxiter_2 = 1000
tol_2 = 1e-9

# I/O
record = False
export = False

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Initial 2D geometry
# ==========================================================================

xyz_origin = [0.0, 0.0, 0.0]

# create skeleton line
line = Line(xyz_origin, add_vectors(xyz_origin, [0.0, length_vault, 0.0]))
arch_nodes, arch_edges = add_arch(network, line, num_segments)

archo_nodes_0 = arch_nodes[:]

# assign supports
network.node_anchor(arch_nodes[0])
network.node_anchor(arch_nodes[-1])

# assign loads
for node in network.nodes_free():
    network.node_load(node, [0.0, 0.0, 0.0])

# assign force densities
for edge in arch_edges:
    network.edge_forcedensity(edge, q0_arch / 2)

# create network in 3d
network_3d = network.copy()
for node in network.nodes_free():
    network_3d.node_load(node, [0.0, 0.0, pz_3d])

# ==========================================================================
# Form-find initial arch skeleton
# ==========================================================================

network = fdm(network)

# ==========================================================================
# Constraint the height of the form-found arch skeleton
# ==========================================================================

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

network_3d = constrained_fdm(network_3d,
                             loss=loss,
                             constraints=constraints,
                             optimizer=SLSQP())

print(f"Starting arch height: {max(network_3d.nodes_attribute(name='z'))}")

# store data
network_3d_old = network_3d.copy()
networks = {-1: network_3d.copy()}
networks_2d = {-1: network.copy()}

# ==========================================================================
# Sequential form-finding
# ==========================================================================

lines = [line] * 2
arches = [arch_nodes] * 2

angles_iter = cycle((angle_base, angle_top))
angle_delta = angle_top - angle_base

for i in range(num_courses):

    print(f"\nCourse {i + 1} /{num_courses}")

    arches_new = []
    lines_new = []

    arch_edges_set = []
    cross_edges_set = []
    arch_nodes_set = []

    cross_edges_dict = {}

    for arch_nodes, line, sign in zip(arches, lines, (-1, 1)):

        line = offset_polyline(line, sign * course_width)
        arch_nodes_new, arch_edges = add_arch(network, line, num_segments)
        cross_edges = connect_arches(network, (arch_nodes, arch_nodes_new))
        arch_nodes = arch_nodes_new

# ==========================================================================
# Define structural system of 2D pattern
# ==========================================================================

        for edge in arch_edges:
            network.edge_forcedensity(edge, q0_arch)

        for edge in cross_edges:
            network.edge_forcedensity(edge, q0_cross)

        for node in arch_nodes[1:-1]:
            network.node_load(node, load=[0.0, 0.0, 0.0])

        # assign supports
        network.node_anchor(arch_nodes[0])
        network.node_anchor(arch_nodes[-1])

# ==========================================================================
# Store data
# ==========================================================================

        arches_new.append(arch_nodes)
        lines_new.append(line)

        arch_nodes_set.extend(arch_nodes)
        arch_edges_set.extend(arch_edges)
        cross_edges_set.extend(cross_edges)
        cross_edges_dict[sign] = cross_edges

# ==========================================================================
# Set new arches as old arches
# ==========================================================================

    arches = arches_new
    lines = lines_new

# ==========================================================================
# Constrained form-finding of 2D pattern - create goals
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

    # transversal planes
    goals_plane = []
    for node in arch_nodes_set:
        if node in anchors:
            continue
        origin = network.node_coordinates(node)
        normal = [0.0, 1.0, 0.0]
        goal = NodePlaneGoal(node, target=(origin, normal), weight=1.0)
        goals_plane.append(goal)

    # middle edge length
    goals_length = []
    for side, _cedges in cross_edges_dict.items():
        edge = _cedges[int(num_segments / 2.)]
        goal = EdgeLengthGoal(edge, target=target_length_2d, weight=1.)
        goals_length.append(goal)

# ==========================================================================
# Constrained form-finding of 2D pattern - define loss function with goals
# ==========================================================================

    loss = Loss(SquaredError(goals=goals_point, name="PointGoal"),
                SquaredError(goals=goals_plane, name="PlaneGoal", alpha=1.),
                SquaredError(goals=goals_length, name="LengthGoal", alpha=1.))

    network = constrained_fdm(network,
                              optimizer=optimizer(),
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol)

    networks_2d[i] = network.copy()

# ==========================================================================
#  Constrained form-finding of 3D pattern - define structural system
# ==========================================================================

    network_3d = network.copy()
    for node in network.nodes_free():
        network_3d.node_load(node, [0.0, 0.0, pz_3d])

    # copy over force densities
    for edge in network_3d_old.edges():
        q = network_3d_old.edge_forcedensity(edge)
        network_3d.edge_forcedensity(edge, q)

    # copy over support positions from old network 3d
    for node in network_3d_old.nodes_supports():
        xyz = network_3d_old.node_coordinates(node)
        network_3d.node_attributes(node, names="xyz", values=xyz)

    # new supports start at the same y coordinate of their old connected anchor
    for node in network_3d.nodes_supports():
        if node not in arch_nodes_set:
            continue
        for nbr in network_3d.neighbors(node):
            if not network_3d.is_node_support(nbr):
                continue
            x, y, z = network_3d.node_coordinates(nbr)
            network_3d.node_attribute(node, "y", y)

# ==========================================================================
#  Constrained form-finding of 3D pattern - set optimization parameters
# ==========================================================================

    parameters = []
    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

    if include_supports_as_params:
        print("Including new courses' support nodes XY coordinates as optimization parameters")
        if i != (num_courses - 1 + 1):
            for node in network.nodes_fixed():
                if node not in arch_nodes_set:
                    continue
                x, y, z = network_3d.node_coordinates(node)
                parameters.append(NodeAnchorXParameter(node, x - xtol, x + xtol))
                parameters.append(NodeAnchorYParameter(node, y - ytol, y + ytol))

# ==========================================================================
#  Constrained form-finding of 2D pattern - define goals
# ==========================================================================

    # horizontal projection goal
    goals_projection = []
    for node in arch_nodes_set:
        if node in anchors:
            continue
        x, y, z = network.node_coordinates(node)
        for goal, coord in ((NodeXCoordinateGoal, x), (NodeYCoordinateGoal, y)):
            goal = goal(node, coord)
            goals_projection.append(goal)

    # best-fit goal
    goals_point = []
    for node in network.nodes_free():
        if node in arch_nodes_set:
            continue
        point = network_3d_old.node_coordinates(node)
        goal = NodePointGoal(node, point, weight=1.)
        goals_point.append(goal)

    # edge length goal
    goals_length = []
    for edge in cross_edges_set:
        goal = EdgeLengthGoal(edge, target=target_length_3d, weight=1.)
        goals_length.append(goal)

    # edge direction goal
    if angle_linear_range:
        d = 0
        if num_courses == 1:
            d = 1
        angle = angle_base + angle_delta * ((i) / ((d + num_courses) - 1))
    else:
        angle = next(angles_iter)
    print(f"Angle goal: {angle:.2f}")

    # node tangent goal
    goals_tangent = []
    for node in arch_nodes_set:
        if node in anchors:
            continue
        goal = NodeTangentAngleGoal(node, vector=[0.0, 0.0, 1.0], target=radians(angle))
        goals_tangent.append(goal)

    goals_direction = []
    edges_middle = []
    for side, _cedges in cross_edges_dict.items():
        edge = _cedges[int(num_segments / 2.)]
        u, v = edge
        edges_middle.append(edge)

        xu, yu, _ = network.node_coordinates(u)  # xyz of first node, assumes it is the lowermost
        u_xyz = [xu, yu, 0.0]

        xv, yv, _ = network.node_coordinates(v)  # xyz of first node, assumes it is the lowermost
        v_xyz = [xv, yv, 0.0]

        angle_rot = angle
        vec = subtract_vectors(v_xyz, u_xyz)
        if dot_vectors(vec, [1., 0., .0]) < 0.0:
            angle_rot = -angle

        vecref = jnp.array([0.0, 0.0, 1.0])

        point = [0.0, 0.0, 1.0]
        end = rotate_points([point], radians(angle_rot), axis=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0]).pop()
        vector = subtract_vectors(end, [0.0, 0.0, 0.0])

        edge = (u, v)
        goal = EdgeDirectionGoal(edge, target=vector, weight=1.)
        goals_direction.append(goal)

# ==========================================================================
#  Constrained form-finding of 2D pattern - set loss function with goals
# ==========================================================================

    loss = Loss(SquaredError(goals=goals_point, name="PointGoal", alpha=10.0),
                SquaredError(goals=goals_projection, name="ProjectionGoal", alpha=1.0),
                SquaredError(goals=goals_direction, name="EdgeDirectionGoal", alpha=10.0),
                SquaredError(goals=goals_length, name="LengthGoal", alpha=0.1),
                SquaredError(goals=goals_tangent, name="NodeTangentAngleGoal", alpha=1.0))

# ==========================================================================
# Solve constrained form-finding in 3D with unconstrained optimization
# ==========================================================================

    opt = optimizer()

    recorder = OptimizationRecorder(opt) if record else None

    network = fdm(network)

    if optimize:
        network_3d = constrained_fdm(network_3d,
                                     optimizer=opt,
                                     parameters=parameters,
                                     loss=loss,
                                     maxiter=maxiter,
                                     tol=tol,
                                     callback=recorder)

# ==========================================================================
# Solve again constrained form-finding in 3D with constrained optimization
# ==========================================================================

    constraints = []

    if add_constraints:
        for u, v in cross_edges_set:
            if u in anchors and v in anchors:
                continue
            edge = (u, v)
            constraint = EdgeLengthConstraint(edge, course_width * (1 - length_delta), course_width * (1 + length_delta))
            constraints.append(constraint)

    if optimize and add_constraints:
        network_3d = constrained_fdm(network_3d,
                                     optimizer=optimizer_2(),
                                     parameters=parameters,
                                     loss=loss,
                                     maxiter=maxiter_2,
                                     constraints=constraints,
                                     tol=tol_2,
                                     callback=None)
# ==========================================================================
# Log and save data
# ==========================================================================

    # Report stats
    network_3d.print_stats()

    networks[i] = network_3d.copy()
    network_3d_old = network_3d

    lengths = []
    for edge in cross_edges_set:
        lengths.append(network_3d.edge_length(*edge))
    print(f"Average edge length in cross edges: {sum(lengths) / len(lengths) }\tInitial course width: {course_width}")

# ==========================================================================
# Plot loss components
# ==========================================================================

    if record:
        print("\n")
        plotter = LossPlotter(loss, network_3d, dpi=150, figsize=(8, 4))
        plotter.plot(recorder.history)
        plotter.show()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

for i, network in networks.items():
    # 3d network
    T = Translation.from_vector([2.2 * i, 0.0, 0.0])
    network = network.transformed(T)
    viewer.add(network,
               edgewidth=(0.01, 0.05),
               loadscale=2.0,
               show_loads=False,
               edgecolor="force")

    # horizontal projection
    network_2d = networks_2d[i].transformed(T)
    viewer.add(network_2d, as_wireframe=True, show_points=False)

    if i >= 0:
        network_old = networks[i - 1]
        viewer.add(network_old.transformed(T), as_wireframe=True, show_points=False)

        # 3d mesh
        cycles = network_find_cycles(network)
        vertices = {vkey: network.node_coordinates(vkey) for vkey in network.nodes()}
        mesh = Mesh.from_vertices_and_faces(vertices, cycles)
        mesh.delete_face(0)
        mesh.cull_vertices()

        viewer.add(mesh, show_points=False, show_lines=False, opacity=0.5)

    # tangent arrows
    if i == num_courses - 1:
        angles_mesh = []
        tangent_angles_mesh = []
        network0_vertices = list(networks[-1].nodes())
        arrows = []
        tangent_arrows = []
        vkeys = []
        for vkey in mesh.vertices():
            xyz = mesh.vertex_coordinates(vkey)
            if xyz[2] < 0.1:
                continue
            if vkey in network0_vertices:
                continue
            normal = mesh.vertex_normal(vkey)

            normal = scale_vector(normal, 0.25)
            z_vector = [0., 0., 1.]
            angle = angle_vectors(z_vector, normal, deg=True)

            ppoint = project_point_plane(add_vectors(xyz, z_vector), (xyz, normal))
            tangent_vector = normalize_vector(subtract_vectors(ppoint, xyz))
            tangent_arrow = Arrow(xyz, scale_vector(tangent_vector, 0.25))
            tangent_arrows.append(tangent_arrow)

            vkeys.append(vkey)

            angles_mesh.append(angle)
            tangent_angles_mesh.append(angle_vectors(z_vector, tangent_vector, deg=True))

            arrow = Arrow(xyz, normal)
            arrows.append(arrow)

        cmap = ColorMap.from_mpl("plasma")
        min_angle = min(tangent_angles_mesh)
        max_angle = max(tangent_angles_mesh)
        print(f"\nTangent angle\tMin: {min_angle:.2f}\tMax: {max_angle:.2f}\tMean: {sum(tangent_angles_mesh)/len(tangent_angles_mesh):.2f}\n")

        for vkey, angle, tangent_angle, arrow, tarrow in zip(vkeys, angles_mesh, tangent_angles_mesh, arrows, tangent_arrows):
            color = cmap(tangent_angle, minval=min_angle, maxval=max_angle)
            viewer.add(tarrow, facecolor=color, show_edges=False, opacity=0.8)
            print(f"node: {vkey}\tangle: {angle:.2f}\ttangent angle: {tangent_angle:.2f}\ttangent angle 2: {90-angle:.2f}")

viewer.show()
