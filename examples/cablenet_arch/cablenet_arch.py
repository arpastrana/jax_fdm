# the essentials
from math import radians

import jax.numpy as jnp
import numpy as np

# compas
from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Polyline
from compas.geometry import Rotation
from compas.geometry import Translation
from compas.geometry import cross_vectors
from compas.geometry import distance_point_point
from compas.geometry import transform_points
from compas.itertools import pairwise

# jax fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeForceGoal
from jax_fdm.goals import EdgesForceEqualGoal
from jax_fdm.goals import MeshSmoothGoal
from jax_fdm.goals import VertexResidualPlaneGoal
from jax_fdm.goals import VertexXCoordinateGoal
from jax_fdm.goals import VertexYCoordinateGoal
from jax_fdm.goals.vertex import VertexGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.losses import PredictionError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import EdgeGroupForceDensityParameter
from jax_fdm.visualization import Viewer

# ==========================================================================
# A custom goal: the highest point of a group of vertices
# ==========================================================================


class VerticesHeightGoal(VertexGoal):
    """
    Drive the highest point of a group of vertices toward a target height.

    Notes
    -----
    The peak is a softmax-weighted average of the vertex heights, a smooth
    stand-in for a hard ``max``. A plain max sends a gradient to a single
    vertex at a time, so the peak can hop between vertices and stall the
    optimizer. The softmax spreads the gradient across the group, weighting
    each vertex by ``exp(z / tau)``, and sharpens toward the true max as the
    temperature ``tau`` shrinks.
    """

    is_aggregate = True

    # tau: ClassVar[float] = 0.1  # softmax temperature, in the height units

    # def prediction(self, eq_state, structure, index):
    #     heights = eq_state.xyz[index, 2]
    #     weights = jax.nn.softmax(heights / self.tau)

    #     return jnp.sum(weights * heights)
    def prediction(self, eq_state, structure, index):
        heights = eq_state.xyz[index, 2]
        return jnp.max(heights)


# ==========================================================================
# A custom goal: fair a chosen group of vertices, boundary left alone
# ==========================================================================


class VerticesSmoothGoal(VertexGoal):
    """
    Fair a group of vertices toward the centroid of their neighbors.

    Notes
    -----
    This is the umbrella-Laplacian fairness energy of the built-in
    ``MeshSmoothGoal``, but averaged only over the vertices we pass as keys.
    A boundary vertex has neighbors on one side only, so its centroid sits
    inward and fairing it cinches the perimeter faces shut. Keying the goal
    on the interior vertices alone fairs the net without dragging the
    boundary in: boundary vertices still feed their neighbors' centroids,
    they are simply never faired themselves.
    """

    is_aggregate = True

    def prediction(self, eq_state, structure, index):
        xyz = eq_state.xyz
        adjacency = structure.adjacency

        num_nbrs = adjacency @ jnp.ones(xyz.shape[0])
        centroids = (adjacency @ xyz) / num_nbrs[:, None]
        fairness = jnp.sum(jnp.square(xyz - centroids), axis=-1) * jnp.square(num_nbrs)

        return jnp.mean(fairness[index])


# ==========================================================================
# Parameters
# ==========================================================================

nx = 20  # number of faces per side
length = 10.0  # side length of the square mesh

rise = 5.0  # height of the arch apex
angle = 20.0  # rotation of the arch about the vertical axis, in degrees

cable_force_pos = 10.0  # target force for the +x cable
cable_force_neg = 5.0  # target force for the -x cable
clearance = 2.5  # minimum walk-under height for the low net vertices

qmin = 1.0
pz = -0.1


# ==========================================================================
# Build the cablenet mesh
# ==========================================================================

mesh = FDMesh.from_meshgrid(length, nx=nx)
mesh.transform(Translation.from_vector([-length / 2.0, -length / 2.0, 0.0]))

# ==========================================================================
# Build a circular arch, no NURBS required
# ==========================================================================

# the arch spans wider than the mesh so its feet reach past the net
half_span = length * 0.75

# a circular arc through (-half_span, 0), (0, rise), (half_span, 0) in the y-z
# plane: center sits on the z axis at z_center, with radius R.
z_center = (rise**2 - half_span**2) / (2.0 * rise)
radius = np.hypot(half_span, z_center)


def arch_point(angle_param):
    """
    Evaluate a point on the circular arch at a sweep angle, then rotate it.
    """
    y = radius * np.cos(angle_param)
    z = z_center + radius * np.sin(angle_param)
    point = [0.0, float(y), float(z)]

    rotation = Rotation.from_axis_and_angle([0.0, 0.0, 1.0], radians(angle))

    return transform_points([point], rotation)[0]


# ==========================================================================
# Pin one column of vertices onto the arch
# ==========================================================================

# the center column of the mesh, sorted along y
arch_vertices = sorted(
    mesh.vertices_where(x=0.0),
    key=lambda v: mesh.vertex_attribute(v, "y"),
)

# sweep from one foot to the other, one angle per vertex
sweep_start = np.arctan2(-z_center, -half_span)
sweep_end = np.arctan2(-z_center, half_span)
sweep = np.linspace(sweep_start, sweep_end, len(arch_vertices))
arch_points = [arch_point(a) for a in sweep]

for vertex, point in zip(arch_vertices, arch_points):
    mesh.vertex_attributes(vertex, "xyz", point)

# the normal to the (rotated) arch plane, for the residual goal
edge_vector = [b - a for a, b in zip(arch_points[0], arch_points[1])]
normal = cross_vectors(edge_vector, [0.0, 0.0, 1.0])

# ==========================================================================
# Assemble the structural system
# ==========================================================================

# supports: the arch column and the four mesh corners
fixed = arch_vertices + list(mesh.vertices_where(vertex_degree=2))
for vertex in fixed:
    mesh.vertex_support(vertex)

# free boundary edges are the perimeter cables we want to equalize
boundary_edges = [
    edge
    for edge in mesh.edges()
    if mesh.is_edge_on_boundary(edge) and not mesh.is_edge_fully_supported(edge)
]

# the supports split the boundary ring into cables: chains of free vertices
# running from one anchor to the next.
fixed_set = set(fixed)
ring = mesh.vertices_on_boundary()
if ring[0] == ring[-1]:
    ring = ring[:-1]

cables = []
chain = []
for vertex in ring:
    if vertex in fixed_set:
        if chain:
            cables.append(chain)
            chain = []
    else:
        chain.append(vertex)
# the ring wraps around, so stitch a trailing chain onto the first one
if chain:
    if ring[0] not in fixed_set and cables:
        cables[0] = chain + cables[0]
    else:
        cables.append(chain)


def cable_score(cable):
    """
    Rank a cable by rest-state length, breaking ties toward positive x.
    """
    points = [mesh.vertex_coordinates(v) for v in cable]
    length = sum(distance_point_point(a, b) for a, b in pairwise(points))
    mean_x = sum(point[0] for point in points) / len(points)

    return length, mean_x


# the two longest cables are the full side edges at x = +/- length/2, one on
# each side, sorted so cable_pos sits at +x and cable_neg at -x
long_cables = sorted(cables, key=cable_score, reverse=True)[:2]
cable_pos, cable_neg = sorted(
    long_cables,
    key=lambda c: sum(mesh.vertex_coordinates(v)[0] for v in c),
    reverse=True,
)

# the +x cable is also the one we lift to the walk-under clearance
cable_midvertex = cable_pos[len(cable_pos) // 2]


def cable_boundary_edges(cable):
    """
    The boundary edges that make up a cable, anchor to anchor.
    """
    cable_set = set(cable)

    return [e for e in boundary_edges if e[0] in cable_set or e[1] in cable_set]


# the boundary edges of each long cable, kept apart for per-cable force targets
cable_pos_edges = cable_boundary_edges(cable_pos)
cable_neg_edges = cable_boundary_edges(cable_neg)

# force densities: stiff boundary, uniform interior, slack under the arch
for edge in mesh.edges():
    q = qmin
    if mesh.is_edge_on_boundary(edge):
        q = 10.0
    elif mesh.is_edge_fully_supported(edge):
        q = 0.01
    mesh.edge_forcedensity(edge, q)

# loads
# for vertex in mesh.vertices_free():
#    mesh.vertex_load(vertex, [0.0, 0.0, pz])

# ==========================================================================
# Form-find a first guess
# ==========================================================================

mesh_guess = fdm(mesh)

forces_guess = [mesh_guess.edge_force(edge) for edge in boundary_edges]
z_guess = mesh_guess.vertex_attribute(cable_midvertex, "z")
print(f"Guess boundary force: min {min(forces_guess):.3f}  max {max(forces_guess):.3f}")
print(f"Guess cable-midpoint height: {z_guess:.3f}")

# ==========================================================================
# Define the optimization problem (three families of goals)
# ==========================================================================

# 1. rotate each arch reaction into the plane of the arch
goals_residual = [
    VertexResidualPlaneGoal(v, target=normal) for v in arch_vertices[1:-1]
]

# 2. equalize each long cable to its own target force
goals_force = [EdgeForceGoal(edge, target=cable_force_pos) for edge in cable_pos_edges]
goals_force += [EdgeForceGoal(edge, target=cable_force_neg) for edge in cable_neg_edges]

# 3. lift the +x cable so its highest point reaches the walk-under height,
# letting the optimizer choose where along the cable that peak lands
goals_height = [VerticesHeightGoal(cable_pos, target=clearance)]

# 4. keep the +x cable plumb, freezing the horizontal position of its vertices
# so it stays a straight run under the arch
goals_plumb = []
for vertex in cable_pos:
    x, y, _ = mesh_guess.vertex_coordinates(vertex)
    goals_plumb.append(VertexXCoordinateGoal(vertex, target=x))
    goals_plumb.append(VertexYCoordinateGoal(vertex, target=y))

# 5. fair the net so the many free force densities settle on a smooth surface,
# smoothing the interior only so the boundary faces are not cinched shut
boundary_vertices = set(mesh.vertices_on_boundary())
interior_vertices = [
    vertex for vertex in mesh.vertices_free() if vertex not in boundary_vertices
]
# goals_smooth = [VerticesSmoothGoal(interior_vertices, target=0.0)]
goals_smooth = [MeshSmoothGoal()]

# 6. even out the internal forces so the net has no stress peaks and can be
# built from a single cable spec, the featured differentiator over a forward
# solver: we optimize the force densities that make the forces uniform
interior_edges = [
    edge
    for edge in mesh.edges()
    if not mesh.is_edge_on_boundary(edge) and not mesh.is_edge_fully_supported(edge)
]
goals_equal = [EdgesForceEqualGoal(interior_edges)]

loss = Loss(
    MeanSquaredError(goals_residual, alpha=10.0, name="ReactionInPlane"),
    MeanSquaredError(goals_force, name="TargetCableForce"),
    # MeanSquaredError(goals_height, alpha=0.1, name="WalkUnderClearance"),
    # MeanSquaredError(goals_plumb, alpha=1.0, name="CableHorizontal"),
    PredictionError(goals_equal, alpha=1.0, name="EqualInternalForce"),
    # smoothing fights force-equalization (drives densities to zero), off for now
    # PredictionError(goals_smooth, alpha=1.0, name="Smoothness"),
    # L2Regularizer(alpha=1.0e-4, name="Regularization"),
)

# each boundary cable gets a single shared force density, so it stays a
# clean, constant-force run; the interior edges keep individual densities
cable_groups = [cable_boundary_edges(cable) for cable in cables]
grouped_edges = {edge for group in cable_groups for edge in group}

parameters = [
    EdgeGroupForceDensityParameter(group, qmin, 100.0) for group in cable_groups
]
parameters += [
    EdgeForceDensityParameter(edge, qmin, 100.0)
    for edge in mesh.edges()
    if not mesh.is_edge_fully_supported(edge) and edge not in grouped_edges
]

# ==========================================================================
# Solve the constrained form-finding problem
# ==========================================================================

mesh_opt = constrained_fdm(
    mesh,
    optimizer=LBFGSB(),
    loss=loss,
    parameters=parameters,
    maxiter=5000,
    tol=1e-6,
)

mesh_opt.print_stats()

forces_pos = [mesh_opt.edge_force(edge) for edge in cable_pos_edges]
forces_neg = [mesh_opt.edge_force(edge) for edge in cable_neg_edges]
z_opt = mesh_opt.vertex_attribute(cable_midvertex, "z")
print(f"Optimized +x cable force: min {min(forces_pos):.3f}  max {max(forces_pos):.3f}")
print(f"Optimized -x cable force: min {min(forces_neg):.3f}  max {max(forces_neg):.3f}")
print(f"Optimized cable-midpoint height: {z_opt:.3f}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer()

# the initial guess as a plain grey wireframe
viewer.add(mesh_guess.copy(cls=Mesh), show_faces=False, edgecolor=(0.4, 0.4, 0.4))

# the optimized cablenet, colored by force density
color_red = Color.red()
color_default = Color.white()
vertex_color = {}
for v in mesh.vertices():
    if v == cable_midvertex:
        vertex_color[v] = color_red
    else:
        vertex_color[v] = color_default

viewer.add(
    mesh_opt,
    fuse=False,
    edgewidth=(0.01, 0.1),
    vertexcolor=vertex_color,
    edgecolor="fd",
    show_vertices=True,
    show_reactions=False,
    show_loads=True,
    loadscale=2.0,
)

# the arch as a cyan polyline
viewer.add(
    Polyline(arch_points),
    linecolor=Color.cyan(),
    lineswidth=3,
    show_points=False,
)

viewer.show()
