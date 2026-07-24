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

segment_length = arch_length / num_segments

xs = [-arch_length / 2.0 + i * segment_length for i in range(num_segments + 1)]
nodes = [[x, 0.0, 0.0] for x in xs]
edges = [(i, i + 1) for i in range(num_segments)]

# ==========================================================================
# Create arch
# ==========================================================================

network = FDNetwork.from_nodes_and_edges(nodes, edges)

# ==========================================================================
# Define structural system
# ==========================================================================

# assign supports
network.node_anchor(key=0)
network.node_anchor(key=num_segments)

# set initial q to all edges
network.edges_forcedensities(q_init, keys=network.edges())

# set initial point loads to all nodes of the network
network.nodes_loads([0.0, 0.0, pz], keys=network.nodes_free())

# ==========================================================================
# Run the force density method
# ==========================================================================

eq_network = fdm(network)

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(show_grid=True)

# equilibrated arch
viewer.add(eq_network, edgewidth=(0.01, 0.1), loadscale=2.0)

# show le crème
viewer.show()
