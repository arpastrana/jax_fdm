import os

from math import fabs

from compas.datastructures import Mesh
from compas.datastructures import network_find_cycles

from compas.colors import Color
from compas.colors import ColorMap
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer
from jax_fdm.visualization import Plotter


name = "rhoen"
plot = False
view = True

target_length = 0.15
threshold = 0.3

# ==========================================================================
# Import network
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}_network0.json"))
network0 = FDNetwork.from_json(FILE_IN)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}_fd.json"))
network_fd = FDNetwork.from_json(FILE_IN)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../../data/json/{name}_optimized_equal.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Visualization
# ==========================================================================

network_fd.edges_forcedensities(0.5)
network_fd = fdm(network_fd)

# ==========================================================================
# Choose network to plot
# ==========================================================================

network = network_fd  # network_fd, network

# ==========================================================================
# Visualization
# ==========================================================================

vertices = {node: network.node_coordinates(node) for node in network.nodes()}
faces = network_find_cycles(network)[1:]
mesh = Mesh.from_vertices_and_faces(vertices, faces)

# ==========================================================================
# Colormaps
# ==========================================================================

# edgecolor = "fd"
cmap = ColorMap.from_mpl("plasma")
lengths_delta = [fabs(network.edge_length(u, v) - target_length) for u, v in network.edges()]
# values = remap_values(lengths_delta, target_min=0.0, target_max=1.0, original_min=0.0, original_max=None)
values = lengths_delta
# print(f"Min: {min(values):.2f} Max: {max(values):.2f} Mean: {(sum(values)/len(values)):.2f}")

edgecolor = {}
for edge, value in zip(network.edges(), values):
    # if value > threshold:
        # value = 1.0
    edgecolor[edge] = cmap(value, minval=0.0, maxval=1.0)

# ==========================================================================
# Plotter
# ==========================================================================

if plot:
    plotter = Plotter(figsize=(16, 9))

    print("Plotting network")

    plotter.add(network,
                edgewidth=1.0,
                edgecolor=edgecolor,
                show_loads=False,
                show_reactions=False,
                show_nodes=False,
                nodesize=50.0,
                )

    plotter.zoom_extents()
    plotter.show()

# ==========================================================================
# Viewer
# ==========================================================================

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    # modify view
    viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out

    viewer.add(network,
               edgewidth=(0.005, 0.02),  # last val: 0.03, 0.08
               edgecolor="force",
               reactioncolor=Color.from_rgb255(0, 150, 10),
               show_reactions=True,
               reactionscale=0.2)

    viewer.add(mesh, show_points=False, show_edges=False, opacity=0.5)

    # reference network
    # viewer.add(network,
    #            as_wireframe=True,
    #            show_points=False,
    #            linewidth=1.0,
    #            color=Color.black().darkened())

    viewer.show()
