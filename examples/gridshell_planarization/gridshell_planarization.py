# the essentials
import numpy as np

# compas
from compas.colors import ColorMap
from compas.datastructures import Mesh
from compas.geometry import Translation

# jax fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import MeshPlanarityGoal
from jax_fdm.goals import MeshSmoothGoal
from jax_fdm.goals import VertexPointGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import VertexSupportXParameter
from jax_fdm.visualization import Viewer

# ==========================================================================
# Parameters
# ==========================================================================

length = 10.0  # side length of the square gridshell
nx = 8  # number of quad faces per side

pz = -1.0  # downward load on every free vertex
q0 = -1.0  # starting force density on the interior edges, negative for compression
q0_boundary = -5.0  # stiffer force density on the free boundary edges, so the
# perimeter tautens and the shell spreads to cover more area underneath

qmin = -50.0  # force density bounds, kept negative to stay compression-only
qmax = -0.01

planarity_weight = 1.0
shape_weight = (
    0.001  # .001, if 0.0 = planarize freely; raise to hold the funicular shape
)
smooth_weight = (
    0.03  # 0.03, fair the net; raise to smooth out jagged, hard-to-build faces
)

# NOTE: MeshSmoothGoal fairs the net toward its flattest state, so on its own it
# collapses the shell to the ground. A nonzero shape_weight anchors the rise, so
# keep shape fidelity on whenever smoothing is on.

pin_side = True  # also pin one full boundary side, not just the four corners
find_supports = True  # let the pinned side's supports slide along y (needs pin_side)
color_log = False  # True: log color scale for planarity; False: linear

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
side_supports = []
if pin_side:
    side = list(mesh.vertices_where(x=-length / 2.0))
    # side_supports = [vertex for vertex in side if vertex not in set(corners)]
    side_supports = [vertex for vertex in side]
    supports += side

for vertex in supports:
    mesh.vertex_support(vertex)

for vertex in mesh.vertices_free():
    mesh.vertex_load(vertex, [0.0, 0.0, pz])

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


maxdev = 0.02  # 2% flatness tolerance: the practical manufacturing limit


def face_flatness(datastructure):
    """
    The flatness of every quad face, keyed by face, relative to the tolerance.

    Uses the COMPAS planar-quad measure: the diagonal gap (the distance between
    a quad's two diagonals, which meet only when it is planar) normalized by the
    average edge length, then divided by ``maxdev``. So the value is a fraction
    of the buildability budget: below 1.0 the panel is within tolerance, above
    1.0 it is too warped to clad with a flat sheet.
    """
    return {
        face: datastructure.face_flatness(face, maxdev=maxdev)
        for face in datastructure.faces()
    }


flatness = face_flatness(shell)
values = list(flatness.values())
rise = max(shell.vertex_attribute(v, "z") for v in shell.vertices_free())
under = 100.0 * sum(1 for f in values if f <= 1.0) / len(values)
print(f"Shell rise: {rise:.3f}")
print(f"Shell face flatness: mean {np.mean(values):.2f}  max {np.max(values):.2f}")
print(f"Faces under the flatness threshold: {under:.0f}%")

# ==========================================================================
# Planarize: find the compression state whose quad faces are flat
# ==========================================================================

# the design variables are the edge force densities, kept negative
parameters = [EdgeForceDensityParameter(edge, qmin, qmax) for edge in mesh.edges()]

# support finding: let each non-corner support on the pinned side slide in and
# out of the edge (its x coordinate becomes a design variable), so the optimizer
# places the supports where they best serve planarity, shape and fairness at once
if find_supports and pin_side:
    parameters += [
        VertexSupportXParameter(vertex, -length / 2.0, length / 2.0)
        for vertex in side_supports
    ]

# the planarity goal drives the mean face non-planarity energy to zero
goals_planar = [MeshPlanarityGoal()]

# optionally hold the funicular shape, trading planarity against shape fidelity
goals_shape = [VertexPointGoal(v, target=shape[v]) for v in mesh.vertices_free()]

# optionally fair the net: planarizing toward a target shape leaves jagged faces
# that are awkward to build, and the smooth goal irons them into buildable panels
goals_smooth = [MeshSmoothGoal()]

error_planar = PredictionError(goals_planar, alpha=planarity_weight, name="Planarity")
error_shape = SquaredError(goals_shape, alpha=shape_weight, name="ShapeFidelity")
error_smooth = PredictionError(goals_smooth, alpha=smooth_weight, name="Smoothness")

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

flatness_planar = face_flatness(shell_planar)
values_planar = list(flatness_planar.values())
compression = all(shell_planar.edge_forcedensity(e) < 0.0 for e in shell_planar.edges())
drift = np.mean(
    [
        np.linalg.norm(
            np.array(shell_planar.vertex_coordinates(v)) - np.array(shape[v]),
        )
        for v in mesh.vertices_free()
    ],
)

under_planar = sum(1 for f in values_planar if f <= 1.0)
pct_planar = 100.0 * under_planar / len(values_planar)
mean_flat, max_flat = np.mean(values_planar), np.max(values_planar)

print()
print(f"Planarized face flatness: mean {mean_flat:.2f}  max {max_flat:.2f}")
n_faces = len(values_planar)
print(f"Panels in tolerance: {under_planar} of {n_faces} ({pct_planar:.0f}%)")
print(f"Still compression-only: {compression}")
print(f"Mean shape drift: {drift:.3f}")

# how far the optimizer slid the supports along the pinned side
if find_supports and pin_side:
    support_moves = [
        np.linalg.norm(
            np.array(shell_planar.vertex_coordinates(v))
            - np.array(shell.vertex_coordinates(v)),
        )
        for v in side_supports
    ]
    mean_move, max_move = np.mean(support_moves), np.max(support_moves)
    print(f"Support travel: mean {mean_move:.3f}  max {max_move:.3f}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(show_grid=True)

# paint each face by its flatness, so the panels hardest to clad stand out. a
# handful of faces are far less flat than the rest, so the log scale spreads the
# near-flat bulk into a legible gradient, where a linear map would wash every
# other face into a single hue.
cmap = ColorMap.from_mpl("plasma")
flatness_shell = face_flatness(shell)


def scale(flatness):
    """
    Map face flatness to the color-scale value, log or linear.
    """
    if color_log:
        eps = 1.0e-4  # floor so exactly-flat faces map to a finite log value
        return {face: np.log10(value + eps) for face, value in flatness.items()}

    return dict(flatness)


# both shells share one color scale, so their faces are directly comparable: the
# initial shell reads hot everywhere, the planarized one mostly cool.
scaled_shell = scale(flatness_shell)
scaled_planar = scale(flatness_planar)
lo = min(min(scaled_shell.values()), min(scaled_planar.values()))
hi = max(max(scaled_shell.values()), max(scaled_planar.values()))


def face_colors(scaled):
    """
    Color every face from the shared planarity scale.
    """
    return {face: cmap((value - lo) / (hi - lo)) for face, value in scaled.items()}


# the initial shell on the left, colored by its (large) face planarity
mesh_shell = shell.copy(Mesh)
mesh_shell.transform(Translation.from_vector([0.0, -1.1 * length, 0.0]))
viewer.add(
    mesh_shell,
    facecolor=face_colors(scaled_shell),
    show_points=False,
    show_lines=True,
    linecolor=(0.2, 0.2, 0.2),
)

# the planarized shell on the right, colored by its (small) face planarity
mesh_planar = shell_planar.copy(Mesh)
viewer.add(
    mesh_planar,
    facecolor=face_colors(scaled_planar),
    show_points=False,
    show_lines=True,
    linecolor=(0.2, 0.2, 0.2),
)

viewer.show()
