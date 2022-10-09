# compas
from compas.geometry import add_vectors
from compas.colors import Color
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

start = [0.0, 0.0, 0.0]
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

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=True)

# equilibrated arch
viewer.add(eq_network,
           edgewidth=(0.01, 0.1),
           edgecolor=Color.teal(),
           reactioncolor=Color.pink())

# reference arch
viewer.add(network,
           as_wireframe=True,
           show_points=False,
           linewidth=2.0)

# show le crème
viewer.show()
