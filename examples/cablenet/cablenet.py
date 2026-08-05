"""
Prestress a square cable-net to meet target forces and target cable lengths.
"""

# jax fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeForceGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer

# ==============================================================================
# Parameters
# ==============================================================================

length = 10.0  # side length of the square cable-net in m
nx = 10  # number of grid cells per side
support_height = 5.0  # lift of the two opposite corner supports in m

force_boundary = 20.0  # target force in the boundary cables in kN
force_interior = 1.0  # target force in the interior cables in kN
length_interior = 1.0  # target length of the interior cables in m

# force density bounds in kN/m, floored well above zero so no cable goes slack
qmin = 0.1
qmax = None

optimizer = LBFGSB()  # optimization algorithm
maxiter = 1000  # optimizer maximum iterations
tol = 1e-8  # optimizer tolerance

# ==============================================================================
# Build a square cable-net
# ==============================================================================

mesh = FDMesh.from_meshgrid(length, nx=nx)

edges_boundary = [edge for edge in mesh.edges() if mesh.is_edge_on_boundary(edge)]
edges_interior = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(edge)]

# ==============================================================================
# Assemble the structural system
# ==============================================================================

# anchor the four corners and lift the diagonal pair to raise the net
corners = list(mesh.vertices_where(vertex_degree=2))
mesh.vertices_supports(corners)
mesh.vertex_attribute(corners[0], "z", support_height)
mesh.vertex_attribute(corners[-1], "z", support_height)

# a force density is a target force divided by a rest length: the boundary
# cables pull harder than the interior ones to keep the perimeter taut
for edge in mesh.edges():
    force = force_boundary if mesh.is_edge_on_boundary(edge) else force_interior
    mesh.edge_forcedensity(edge, force / mesh.edge_length(edge))

# ==============================================================================
# Form-find the tensile net
# ==============================================================================

cablenet = fdm(mesh)

# ==============================================================================
# Print results
# ==============================================================================

print(f"Target boundary force: {force_boundary}, interior length: {length_interior}")

extra_stats = {
    "Boundary force": [cablenet.edge_force(e) for e in edges_boundary],
    "Interior length": [cablenet.edge_length(e) for e in edges_interior],
}

cablenet.print_stats(extra_stats)

# ==============================================================================
# Prestress: hit the target forces and iron out the interior cables
# ==============================================================================

# the design variables are the edge force densities, bounded to stay in tension
parameters = [EdgeForceDensityParameter(edge, qmin, qmax) for edge in mesh.edges()]

# pull the boundary cables to their target force, and even out the interior ones
goals_force = [EdgeForceGoal(edge, target=force_boundary) for edge in edges_boundary]
goals_length = [EdgeLengthGoal(edge, target=length_interior) for edge in edges_interior]

loss = Loss(
    MeanSquaredError(goals_force, name="BoundaryForce"),
    MeanSquaredError(goals_length, name="InteriorLength"),
)

# the recorder stores the parameters per iteration to chart the loss afterwards
recorder = OptimizationRecorder(optimizer)

cablenet_prestressed = constrained_fdm(
    mesh,
    optimizer=optimizer,
    loss=loss,
    parameters=parameters,
    maxiter=maxiter,
    tol=tol,
    callback=recorder,
)

# ==============================================================================
# Print results
# ==============================================================================

extra_stats = {
    "Boundary force": [cablenet_prestressed.edge_force(e) for e in edges_boundary],
    "Interior length": [cablenet_prestressed.edge_length(e) for e in edges_interior],
}

cablenet_prestressed.print_stats(extra_stats)

# ==============================================================================
# Plot loss components
# ==============================================================================

plotter = LossPlotter(loss, mesh, dpi=150, figsize=(8, 4))
plotter.plot(recorder.history)
plotter.show()

# ==============================================================================
# Visualization
# ==============================================================================

viewer = Viewer(width=1200, height=800, show_grid=True)

viewer.add(
    cablenet_prestressed,
    edgecolor="force",
    show_vertices=True,
    reactionscale=0.05,
    faceopacity=0.5,
)

viewer.show()
