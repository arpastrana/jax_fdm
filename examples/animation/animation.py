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
from compas.geometry import scale_vector

# jax_fdm
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import network_update

from jax_fdm.visualization import Viewer


# ==========================================================================
# Read in optimization history
# ==========================================================================

name = "butt"

modify_view = True
show_grid = False
camera_zoom = -50  # -35 for monkey saddle, 0 for pringle, 14 for dome, -70 for butt

decimate = False
decimate_step = 0

interval = 50  # 50
timeout = None
fps = 8

animate = True
rotate_while_animate = False
save = True


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


def lines_draw(network, func_name, scale):
    lines = {}
    for node in network.nodes():
        pt = network.node_coordinates(node)
        vector = getattr(network, func_name)(node)
        vector = scale_vector(vector, scale)
        line = Line(pt, add_vectors(pt, vector))
        lines[node] = line
    return lines


def loads_draw(network, scale=1.0):
    return lines_draw(network, "node_load", scale)


def residuals_draw(network, scale=1.0):
    return lines_draw(network, "node_residual", scale)


def lines_update(lines, network, func_name, scale):
    for node, line in lines.items():
        pt = network.node_coordinates(node)
        line.start = pt
        line.end = add_vectors(pt, scale_vector(getattr(network, func_name)(node), scale))


def loads_update(loads, network, scale=1.0):
    lines_update(loads, network, "node_load", scale)


def residuals_update(residuals, network, scale=1.0):
    lines_update(residuals, network, "node_residual", scale)


# ==========================================================================
# Read in force density network
# ==========================================================================

HERE = os.path.join(os.path.dirname(__file__), "../../data/json/")
FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_base.json"))
network0 = FDNetwork.from_json(FILE_IN)
model = EquilibriumModel(network0)
network = fdm(network0)

# ==========================================================================
# Read in optimization history
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_history.json"))
recorder = OptimizationRecorder.from_json(FILE_IN)

# ==========================================================================
# Visualization
# ==========================================================================

# instantiate viewer
viewer = Viewer(width=1600, height=900, show_grid=show_grid)

# modify view
if modify_view:
    viewer.view.camera.target[2] -= 5.0
    viewer.view.camera.zoom(camera_zoom)  # number of steps, negative to zoom out
    viewer.view.camera.rotation[2] = 0.8 * pi / 3  # set rotation around z axis to zero
    # viewer.view.camera.rotation_delta = (2 / 3) * pi / len(recorder.history)  # set rotation around z axis to zero

# draw network
network_obj = viewer.add(network,
                         as_wireframe=True,
                         show_points=False,
                         linewidth=10.0,
                         color=Color.grey().darkened())

# draw supports
# for node in network.nodes_supports():
#     x, y, z = network.node_coordinates(node)
#     viewer.add(Point(x, y, z), color=Color.green(), size=20)

# draw loads
load_scale = 4
loads = loads_draw(network, load_scale)
for load in loads.values():
    viewer.add(load, linewidth=2.0, color=Color.green().darkened())

# draw residual forces
residuals = residuals_draw(network)
for residual in residuals.values():
    viewer.add(residual, linewidth=8.0, color=Color.pink())

# warm start model
_ = model(np.array(recorder.history[0]))

# decimate
if decimate:
    history = recorder.history[::decimate_step]
    recorder.history = history

# create update function
if animate:
    config_animate = {"interval": interval,
                      "timeout": timeout,
                      "frames": len(recorder.history),
                      "record": save,
                      "record_fps": fps,
                      "record_path": f"temp/{name}.gif"}

    @viewer.on(**config_animate)
    def wiggle(f):

        print(f"Current frame: {f + 1}/{len(recorder.history)}")
        q = np.array(recorder.history[f])
        eqstate = model(q)

        # update network
        network_update(network, eqstate)
        network_obj.linecolors = edge_colors(network)

        # update loads
        loads_update(loads, network, load_scale)

        # update residual forces
        residuals_update(residuals, network)

        for _, obj in viewer.view.objects.items():
            obj.update()

        if rotate_while_animate:
            viewer.view.camera.rotate(dx=1, dy=0)

        # update all buffer objects in the view
        # for artist in viewer.artists:
        #   artist.update(network)
        #   for obj in artist.objects:
        #       obj.update()

# show le crème
viewer.show()
