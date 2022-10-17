# compas
from compas.geometry import add_vectors
from compas.geometry import Polyline

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm

from jax_fdm.visualization import Viewer

# ==========================================================================
# Initial parameters
# ==========================================================================

arch_length = 5.0
num_segments = 10
q_init = -1
pz = -0.2

# ==========================================================================
# Create the geometry of an arch
# ==========================================================================

start = [-arch_length / 2.0, 0.0, 0.0]
end = add_vectors(start, [arch_length, 0.0, 0.0])
curve = Polyline([start, end])
points = curve.divide_polyline(num_segments)
lines = Polyline(points).lines

# ==========================================================================
# Create arch
# ==========================================================================

network = FDNetwork.from_lines(lines)

# ==========================================================================
# Define structural system
# ==========================================================================

# assign supports
network.node_support(key=0)
network.node_support(key=len(points) - 1)

# set initial q to all edges
network.edges_forcedensities(q_init, keys=network.edges())

# set initial point loads to all nodes of the network
network.nodes_loads([0.0, 0.0, pz], keys=network.nodes_free())

# ==========================================================================
# Run thee force density method
# ==========================================================================

eq_network = fdm(network)

eq_network.print_stats()

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import NodeResidualForceGoal, NodeLineGoal
from jax_fdm.optimization import SLSQP
from jax_fdm.losses import RootMeanSquaredError, Loss
from compas.geometry import Line, add_vectors


goals = []

goal = NodeResidualForceGoal(0, 0.3)
goals.append(goal)

for node in network.nodes_free():
    xyz = network.node_coordinates(node)
    line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
    goal = NodeLineGoal(node, line)
    goals.append(goal)

# segment_length = arch_length / num_segments
# target_length = segment_length * 1.5
# print(segment_length, target_length)

# for edge in network.edges():
#     goal = EdgeLengthGoal(edge, target=target_length)
#     goals.append(goal)

loss = Loss(RootMeanSquaredError(goals))

c_network = constrained_fdm(network, SLSQP(), loss, bounds=(None, 0.0))

c_network.print_stats()
# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=True)

# equilibrated arch
viewer.add(eq_network, as_wireframe=True, show_points=False)
viewer.add(c_network, edgecolor="force", edgewidth=(0.01, 0.1))

# show le crème
viewer.show()
