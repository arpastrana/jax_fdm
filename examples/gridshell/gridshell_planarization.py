# the essentials
import numpy as np

# compas
from compas.colors import ColorMap
from compas.geometry import Translation

# jax fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import MeshPlanarityGoal
from jax_fdm.goals import MeshSmoothGoal
from jax_fdm.goals import VertexPointGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.losses import PredictionError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import VertexSupportXParameter
from jax_fdm.visualization import Viewer

# ==========================================================================
# Helper functions
# ==========================================================================


def face_flatness(datastructure, maxdev=0.01):
    """
    The flatness of every quad face, keyed by face, relative to the tolerance.

    Uses the COMPAS planar-quad measure: the diagonal gap (the distance between
    a quad's two diagonals, which meet only when it is planar) normalized by the
    average edge length, then divided by ``maxdev``. So the value is a fraction
    of the buildability budget: below 1.0 the panel is within tolerance, above
    1.0 it is too warped to clad with a flat sheet. Default tolerance is 1%.
    """
    return {
        face: datastructure.face_flatness(face, maxdev=maxdev)
        for face in datastructure.faces()
    }


def face_colors(scaled, lo, hi):
    """
    Color every face from the shared planarity scale.
    """
    cmap = ColorMap.from_mpl("plasma")
    return {face: cmap((value - lo) / (hi - lo)) for face, value in scaled.items()}


def distance_vertices(mesh, mesh_target, vertices):
    """
    The distance between each vertex and the target.
    """
    distances = []
    for vertex in vertices:
        xyz = np.array(mesh.vertex_coordinates(vertex))
        xyz_target = np.array(mesh_target.vertex_coordinates(vertex))
        distance = np.linalg.norm(xyz - xyz_target)
        distances.append(distance)
    return distances


# ==========================================================================
# Parameters
# ==========================================================================

length = 10.0  # side length of the square gridshell
nx = 8  # 8, 12 number of quad faces per side

q0 = -1.0  # starting force density on the interior edges, negative for compression
q0_boundary = -5.0  # stiffer force density on the free boundary edges for tautness

# force density bounds, kept negative to stay compression-only
qmin = -50.0
qmax = -0.01

# error weights
planarity_weight = 1.0
smooth_weight = 0.0  # smooth the shape, raise to iron out jagged faces
shape_weight = 0.0  # if 0.0 = modify freely, raise to hold the funicular shape

pin_side = True  # also pin one full boundary side, not just the four corners
find_supports = False  # let the pinned side's supports slide along x (needs pin_side)

# ==========================================================================
# Build a square quad gridshell
# ==========================================================================

mesh = FDMesh.from_meshgrid(length, nx=nx)
mesh.transform(Translation.from_vector([-length / 2.0, -length / 2.0, 0.0]))

# ==========================================================================
# Assemble the structural system: a corner-supported compression shell
# ==========================================================================

# pin the four corners, and optionally one full boundary side (at x = -length/2)
corners = list(mesh.vertices_where(vertex_degree=2))
supports = list(corners)
side = []
if pin_side:
    side = list(mesh.vertices_where(x=-length / 2.0))
    supports += side

for vertex in supports:
    mesh.vertex_support(vertex)

for vertex in mesh.vertices_free():
    mesh.vertex_load(vertex, [0.0, 0.0, -1.0])

# negative force densities put the whole shell in compression: a stiffer value
# on the free boundary edges tautens the perimeter so the shell spreads wider
for edge in mesh.edges():
    if mesh.is_edge_on_boundary(edge) and not mesh.is_edge_fully_supported(edge):
        mesh.edge_forcedensity(edge, q0_boundary)
    else:
        mesh.edge_forcedensity(edge, q0)

# ==========================================================================
# Form-find the compression shell
# ==========================================================================

shell = fdm(mesh)

# record the funicular shape, in case we want to hold onto it while planarizing
shape = {vertex: shell.vertex_coordinates(vertex) for vertex in mesh.vertices_free()}

# stats
flatness = face_flatness(shell)
values = list(flatness.values())
rise = max(shell.vertex_attribute(vertex, "z") for vertex in mesh.vertices_free())
under = 100.0 * sum(1 for f in values if f <= 1.0) / len(values)
print(f"Shell rise: {rise:.3f}")
print(f"Shell face flatness: mean {np.mean(values):.2f}  max {np.max(values):.2f}")
print(f"Faces under the flatness threshold: {under:.0f}%")

# ==========================================================================
# Planarize: find the compression state whose quad faces are flat
# ==========================================================================

# the design variables are the edge force densities
parameters = [EdgeForceDensityParameter(edge, qmin, qmax) for edge in mesh.edges()]

# support finding: let each non-corner support on the pinned side slide in and out
if find_supports and pin_side:
    parameters_supports = []
    for vertex in side:
        x = shell.vertex_coordinates(vertex)[0]
        xtol = 0.1 * length
        parameter = VertexSupportXParameter(vertex, x - xtol, x + xtol)
        parameters_supports.append(parameter)
    parameters += parameters_supports

# the planarity goal drives the faces non-planarity to zero
goals_planar = [MeshPlanarityGoal()]

# smoothing the shell
goals_smooth = [MeshSmoothGoal()]

# hold the starting funicular shape
goals_shape = [
    VertexPointGoal(vertex, target=shape[vertex]) for vertex in mesh.vertices_free()
]

# assemble the errors
error_planar = PredictionError(goals_planar, alpha=planarity_weight, name="Planarity")
error_smooth = PredictionError(goals_smooth, alpha=smooth_weight, name="Smoothness")
error_shape = MeanSquaredError(goals_shape, alpha=shape_weight, name="ShapeFidelity")

# keep only the goals with a positive weight
candidates = [
    (error_planar, planarity_weight),
    (error_shape, shape_weight),
    (error_smooth, smooth_weight),
]

loss = Loss(*[error for error, weight in candidates if weight > 0.0])

shell_planar = constrained_fdm(
    mesh,
    optimizer=LBFGSB(),
    loss=loss,
    parameters=parameters,
    maxiter=5000,
    tol=1e-8,
)

# print the stats
shell_planar.print_stats()

flatness_planar = face_flatness(shell_planar)
values_planar = list(flatness_planar.values())
drift = np.mean(distance_vertices(shell_planar, shell, mesh.vertices_free()))

under_planar = sum(1 for f in values_planar if f <= 1.0)
pct_planar = 100.0 * under_planar / len(values_planar)
mean_flat, max_flat = np.mean(values_planar), np.max(values_planar)
n_faces = len(values_planar)

print()
print(f"Planarized face flatness: mean {mean_flat:.2f}  max {max_flat:.2f}")
print(f"Panels in tolerance: {under_planar} of {n_faces} ({pct_planar:.0f}%)")
print(f"Mean shape drift: {drift:.3f}")

# how far the optimizer slid the supports along the pinned side
if find_supports and pin_side:
    support_moves = distance_vertices(shell_planar, shell, side)
    mean_move, max_move = np.mean(support_moves), np.max(support_moves)
    print(f"Support travel: mean {mean_move:.3f}  max {max_move:.3f}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(show_grid=True)

# paint each face by its flatness, so the panels hardest to clad stand out
flatness_shell = face_flatness(shell)

# both shells share one color scale to compare their flatness directly
lo = min(*flatness_shell.values(), *flatness_planar.values())
hi = max(*flatness_shell.values(), *flatness_planar.values())

# the initial shell placed on the left
shell.transform(Translation.from_vector([0.0, -1.1 * length, 0.0]))
viewer.add(
    shell,
    facecolor=face_colors(flatness_shell, lo, hi),
    faceopacity=1.0,
    show_vertices=True,
    show_loads=False,
    show_reactions=False,
)

# the planarized shell
viewer.add(
    shell_planar,
    facecolor=face_colors(flatness_planar, lo, hi),
    faceopacity=1.0,
    show_vertices=True,
    show_loads=False,
    show_reactions=False,
)

viewer.show()
