# plotting
import os

import matplotlib.pyplot as plt
from compas_viewer.config import Config
from compas_viewer.config import RendererConfig
from compas_viewer.config import WindowConfig

# compas
from compas.datastructures import Network
from compas.geometry import Polyline
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
# Build the truss: top chord, bottom chord, and struts
# ==========================================================================

top_line = Polyline([[-length / 2.0, 0.0, 0.0], [length / 2.0, 0.0, 0.0]])
points = top_line.divide(num_segments)
network = FDNetwork.from_lines(Polyline(points).lines)

# supports at both ends, a downward load on every free top node, compressive q
network.node_anchor(key=0)
network.node_anchor(key=num_segments)
network.nodes_loads([0.0, 0.0, -0.2], keys=network.nodes_free())

for edge in network.edges():
    network.edge_forcedensity(edge, -1.0)

# keep a handle on the top chord before adding the rest
top_nodes = sorted(network.nodes())
top_edges = list(network.edges())
top_nodes_free = list(network.nodes_free())

# bottom-chord nodes, offset below the top chord
bottom_line = Polyline([[-length / 2.0, depth, 0.0], [length / 2.0, depth, 0.0]])
offsets = bottom_line.divide(num_segments)
bottom_nodes = [network.add_node(x=p[0], y=p[1], z=p[2]) for p in offsets[1:-1]]

# bottom-chord edges (tension), spanning support to support
bottom_edges = []
for u, v in pairwise([0] + bottom_nodes + [num_segments]):
    edge = network.add_edge(u, v)
    network.edge_forcedensity(edge, 2.0)
    bottom_edges.append(edge)

# struts tying each bottom node to a top node
for u, v in zip(bottom_nodes, top_nodes_free):
    network.add_edge(u, v)
    network.edge_forcedensity((u, v), -0.5)

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

# compas_viewer's camera constructor hardcodes the perspective view, ignoring
# the configured one; reset the camera so it picks up renderer.view ("front").
viewer.renderer.camera.reset_position()

# modify view
viewer.renderer.camera.target = (0.0, 0.0, -0.5)
viewer.renderer.camera.position = (0.0, -2.6, -0.5)

# initial guess as plain wireframe
viewer.add(network_guess.copy(cls=Network))

# optimized truss colored by member force
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
