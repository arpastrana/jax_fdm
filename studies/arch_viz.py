# math is always a good idea
from math import pi

# compas
from compas.geometry import add_vectors
from compas.geometry import Polyline
from compas.geometry import Rotation

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm

from jax_fdm.visualization import Plotter


# ==========================================================================
# Initial parameters
# ==========================================================================

arch_length = 5.0
num_segments = 10
q_init = -1
py = -0.2

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
network.node_anchor(key=0)
network.node_anchor(key=len(points) - 1)

# set initial q to all edges
network.edges_forcedensities(q_init, keys=network.edges())

# set initial point loads to all nodes of the network
network.nodes_loads([0.0, py, 0.0], keys=network.nodes_free())

# ==========================================================================
# Run thee force density method
# ==========================================================================

eq_network = fdm(network)

# ==========================================================================
# Visualization
# ==========================================================================

plotter = Plotter(dpi=150)

plotter.add(eq_network,
            show_nodes=True,
            nodesize=0.4,
            edgewidth=(0.5, 5),
            reactionscale=1.0,
            loadscale=2.0)

# show le crème
plotter.zoom_extents()
if plot_save:
    print("Saving")
    plotter.save(filepath, dpi=300, bbox_inches="tight", transparent=True)

plotter.show()
