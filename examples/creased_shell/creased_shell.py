# the essentials
import os

from scipy.spatial.distance import directed_hausdorff

# compas
from compas.colors import Color
from compas.datastructures import Network
from compas.geometry import Line

# jax fdm
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import NodePointGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import RootMeanSquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.visualization import Viewer

# ==========================================================================
# Import network
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, "../../data/json/creased_shell.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Record the target shape
# ==========================================================================

targets = {node: network.node_coordinates(node) for node in network.nodes()}

# ==========================================================================
# Define structural system
# ==========================================================================

supports = [node for node in network.nodes() if network.is_leaf(node)]
network.nodes_supports(supports)
network.nodes_loads([0.0, 0.0, -0.2], keys=network.nodes_free())
network.edges_forcedensities(q=-1.0)

# ==========================================================================
# Define optimization parameters
# ==========================================================================

parameters = []
for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, -20.0, 0.0)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

goals = []
for node in network.nodes_free():
    goal = NodePointGoal(node, target=targets[node])
    goals.append(goal)

# ==========================================================================
# Combine error functions into a loss function
# ==========================================================================

loss = Loss(RootMeanSquaredError(goals))

# ==========================================================================
# Form-find the network for a first guess
# ==========================================================================

network_guess = fdm(network)

print(f"Load path: {round(network_guess.loadpath(), 3)}")

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

network_matched = constrained_fdm(
    network,
    optimizer=LBFGSB(),
    loss=loss,
    parameters=parameters,
    maxiter=1000,
    tol=1e-6,
)

# ==========================================================================
# Hausdorff distance
# ==========================================================================

matched = [network_matched.node_coordinates(n) for n in network_matched.nodes()]
target = [targets[n] for n in network_matched.nodes()]

forward = directed_hausdorff(matched, target)[0]
backward = directed_hausdorff(target, matched)[0]
hausdorff = max(forward, backward)
print(f"Hausdorff distance: {hausdorff:.3f}")

# ==========================================================================
# Report stats
# ==========================================================================

network_matched.print_stats()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer()

# modify view
viewer.renderer.camera.target = (1.5, 0.0, -2.0)
viewer.renderer.camera.position = (20.0, -16.0, 18.5)

# optimized network
viewer.add(
    network_matched,
    edgewidth=(0.1, 0.2),
    edgecolor="fd",
    show_reactions=False,
    show_loads=False,
    show_nodes=True,
    nodesize=0.2,
)

# reference network as plain geometry
viewer.add(network.copy(cls=Network), show_points=False, color=Color.grey())

# draw lines to target
for node in network_matched.nodes():
    line = Line(network_matched.node_coordinates(node), targets[node])
    viewer.add(line, color=Color.red())

# show le crème
viewer.show()
