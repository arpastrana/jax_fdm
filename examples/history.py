# the essentials
import os
from math import fabs
from math import pi
import numpy as np

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors

# dfdm
from dfdm.optimization import OptimizationRecorder
from dfdm.datastructures import FDNetwork
from dfdm.equilibrium import EquilibriumModel
from dfdm.equilibrium import fdm
from dfdm.equilibrium import network_update

# compas view
from compas_view2.app import App


# ==========================================================================
# Read in optimization history
# ==========================================================================

name = "monkey_saddle"
interval = 50
animate = True
record = False

# ==========================================================================
# Helper functions
# ==========================================================================

def edge_colors(network, cmap_name="viridis"):
    cmap = ColorMap.from_mpl(cmap_name)
    fds = [fabs(network.edge_forcedensity(edge)) for edge in network.edges()]
    colors = {}
    for edge in network.edges():
        fd = fabs(network.edge_forcedensity(edge))
        try:
            ratio = (fd - min(fds)) / (max(fds) - min(fds))
        except ZeroDivisionError:
            ratio = 1.0
        colors[edge] = cmap(ratio)
    return colors


def lines_draw(network, func_name):
    lines = {}
    for node in network.nodes():
        pt = network.node_coordinates(node)
        vector = getattr(network, func_name)(node)
        line = Line(pt, add_vectors(pt, vector))
        lines[node] = line
    return lines


def loads_draw(network):
    return lines_draw(network, "node_load")


def residuals_draw(network):
    return lines_draw(network, "node_residual")


def lines_update(lines, network, func_name):
    for node, line in lines.items():
        pt = network.node_coordinates(node)
        line.start = pt
        line.end = add_vectors(pt, getattr(network, func_name)(node))


def loads_update(loads, network):
    lines_update(loads, network, "node_load")


def residuals_update(residuals, network):
    lines_update(residuals, network, "node_residual")


# ==========================================================================
# Read in force density network
# ==========================================================================

HERE = os.path.join(os.path.dirname(__file__), "../data/json/")
FILE_IN= os.path.abspath(os.path.join(HERE, f"{name}_base.json"))
network0 = FDNetwork.from_json(FILE_IN)
model = EquilibriumModel(network0)
network = fdm(network0)

# ==========================================================================
# Read in optimization history
# ==========================================================================

FILE_IN= os.path.abspath(os.path.join(HERE, f"{name}_history.json"))
recorder = OptimizationRecorder.from_json(FILE_IN)

# ==========================================================================
# Visualization
# ==========================================================================

# instantiate viewer
viewer = App(width=1600, height=900, show_grid=True)

# modify view
viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.view.camera.rotation[2] = 2 * pi / 3  # set rotation around z axis to zero
viewer.view.camera.rotation_delta = (2 / 3) * pi / len(recorder.history)  # set rotation around z axis to zero

# draw network
viewer.add(network.copy(), show_points=False, linewidth=1.0, color=Color.grey())
network_obj = viewer.add(network, show_points=False, linewidth=5.0, color=Color.grey().darkened())

# draw supports
for node in network.nodes_supports():
    x, y, z = network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=20)

# draw loads
loads = loads_draw(network)
for load in loads.values():
    viewer.add(load, linewidth=4.0, color=Color.green().darkened())

# draw residual forces
residuals = residuals_draw(network)
for residual in residuals.values():
    viewer.add(residual, linewidth=4.0, color=Color.pink())

# create update function
if animate:
    @viewer.on(interval=interval, frames=len(recorder.history), record=record, record_path=f"temp/{name}.gif")
    def wiggle(f):
        q = np.array(recorder.history[f])
        eqstate = model(q)

        # update network
        network_update(network, eqstate)
        network_obj.linecolors = edge_colors(network)

        # update loads
        loads_update(loads, network)

        # update residual forces
        residuals_update(residuals, network)

        # update all buffer objects in the view
        for _, obj in viewer.view.objects.items():
            obj.update()

        viewer.view.camera.rotate(dx=1, dy=0)



# show le crème
viewer.show()
