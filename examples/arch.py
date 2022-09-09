# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import length_vector

# visualization
from compas_view2.app import App

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm

# ==========================================================================
# Initial parameters
# ==========================================================================

arch_length = 5.0
num_segments = 10
q_init = -1
pz = -0.1

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

# viewer = App(width=1600, height=900, show_grid=True)

# # equilibrated arch
# viewer.add(eq_network,
#            show_vertices=True,
#            pointsize=12.0,
#            show_edges=True,
#            linecolor=Color.teal(),
#            linewidth=5.0)

# # reference arch
# viewer.add(network, show_points=False, linewidth=4.0)

# for node in eq_network.nodes():

#     pt = eq_network.node_coordinates(node)

#     # draw lines betwen subject and target nodes
#     target_pt = network.node_coordinates(node)
#     viewer.add(Line(target_pt, pt))

#     # draw residual forces
#     residual = eq_network.node_residual(node)

#     if length_vector(residual) < 0.001:
#         continue

#     residual_line = Line(pt, add_vectors(pt, residual))
#     viewer.add(residual_line,
#                linewidth=4.0,
#                color=Color.pink())

# # draw applied loads
# for node in eq_network.nodes():
#     pt = eq_network.node_coordinates(node)
#     load = network.node_load(node)
#     viewer.add(Line(pt, add_vectors(pt, load)),
#                linewidth=4.0,
#                color=Color.green().darkened())

# # draw supports
# for node in eq_network.nodes_supports():
#     x, y, z = eq_network.node_coordinates(node)
#     viewer.add(Point(x, y, z), color=Color.green(), size=20)

# # show le crème
# viewer.show()
