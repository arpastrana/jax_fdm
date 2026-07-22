# plotting
import os

import matplotlib.pyplot as plt
from compas_viewer.config import Config
from compas_viewer.config import RendererConfig
from compas_viewer.config import WindowConfig

# compas
from compas.datastructures import Network
from compas.itertools import pairwise

# jax fdm
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeForceGoal
from jax_fdm.goals import NodesColinearGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.losses import PredictionError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.visualization import Viewer

# ==========================================================================
# Parameters
# ==========================================================================

length = 5.0
num_segments = 11
depth = -2.0
target_force = 1.5

# ==========================================================================
# Create the spaced points along x
# ==========================================================================

step = length / num_segments
xs = [-length / 2.0 + i * step for i in range(num_segments + 1)]

top_points = [[x, 0.0, 0.0] for x in xs]  # with supports at both ends
bottom_points = [[x, depth, 0.0] for x in xs[1:-1]]  # without supports

# ==========================================================================
# Build the truss network: top chord, bottom chord, and struts
# ==========================================================================

# create an empty network
network = FDNetwork()

# add the nodes
top_nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in top_points]
top_nodes_free = top_nodes[1:-1]
bottom_nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in bottom_points]

# top chord, bottom chord, and struts between them
bottom_chord = [top_nodes[0]] + bottom_nodes + [top_nodes[-1]]
top_edges = [network.add_edge(u, v) for u, v in pairwise(top_nodes)]
bottom_edges = [network.add_edge(u, v) for u, v in pairwise(bottom_chord)]
strut_edges = [network.add_edge(u, v) for u, v in zip(bottom_nodes, top_nodes_free)]

# ==========================================================================
# Assemble the structural system
# ==========================================================================

# supports at both ends
network.node_support(top_nodes[0])
network.node_support(top_nodes[-1])

# downard load on every free top node
for node in top_nodes_free:
    network.node_load(node, [0.0, 0.0, -0.2])

# top chord in compression
for edge in top_edges:
    network.edge_forcedensity(edge, -1.0)

# bottom chord in tension
for edge in bottom_edges:
    network.edge_forcedensity(edge, 2.0)

# struts mildly in compression
for edge in strut_edges:
    network.edge_forcedensity(edge, -0.5)

# ==========================================================================
# Form-find the truss for a first guess
# ==========================================================================

network_guess = fdm(network)

forces_guess = [network_guess.edge_force(edge) for edge in bottom_edges]
print(f"Guess force: min {min(forces_guess):.3f} max {max(forces_guess):.3f}")

# ==========================================================================
# Define the equal-force problem (two goals)
# ==========================================================================

# aim every bottom edge at the same force, so they are equal by construction
goals_force = [EdgeForceGoal(edge, target=target_force) for edge in bottom_edges]

# keep the top chord straight
goals_colinear = [NodesColinearGoal(key=top_nodes)]

loss = Loss(
    MeanSquaredError(goals_force, name="ForceTarget"),
    PredictionError(goals_colinear, name="ColinearTopChord"),
)

# ==========================================================================
# Solve the constrained form-finding problem
# ==========================================================================

optimizer = LBFGSB()
recorder = OptimizationRecorder(optimizer)

network_equalforce = constrained_fdm(
    network,
    optimizer=optimizer,
    loss=loss,
    maxiter=10000,
    tol=1e-9,
    callback=recorder,
)

forces = [network_equalforce.edge_force(edge) for edge in bottom_edges]
print(f"Optimized bottom-chord force: min {min(forces):.3f}  max {max(forces):.3f}")

# ==========================================================================
# Bottom-chord force plot (before vs after)
# ==========================================================================

segments = range(1, len(bottom_edges) + 1)

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(segments, forces_guess, "o-", label="Before (FDM guess)")
ax.plot(segments, forces, "s-", label="After (Constrained FDM)")
ax.axhline(target_force, ls="--", color="grey", label=f"Target = {target_force}")
ax.set_xlabel("Bottom-chord segment")
ax.set_ylabel("Axial force")
ax.legend()
fig.tight_layout()

HERE = os.path.dirname(__file__)
IMAGES = os.path.join(HERE, "../../docs/assets/images")
fig.savefig(os.path.abspath(os.path.join(IMAGES, "truss_equalforce_force_plot.png")))
plt.show()

# ==========================================================================
# Visualization
# ==========================================================================

# default the scene to a front view for screenshots
config = Config(
    window=WindowConfig(width=1000, height=600),
    renderer=RendererConfig(show_grid=False, rendermode="lighted", view="front"),
)
viewer = Viewer(config=config)

# reset the camera so it picks up renderer.view ("front").
viewer.renderer.camera.reset_position()

# modify view
viewer.renderer.camera.target = (0.0, 0.0, -0.5)
viewer.renderer.camera.position = (0.0, -2.6, -0.5)

# initial guess as plain wireframe
viewer.add(network_guess.copy(cls=Network))

# optimized truss colored by member force with variable edge width
viewer.add(
    network_equalforce,
    edgewidth=(0.01, 0.05),
    edgecolor="force",
    show_nodes=True,
    show_loads=True,
    nodesize=0.05,
    show_reactions=False,
    loadscale=2.0,
)

# show le crème
viewer.show()
