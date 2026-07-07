# the essentials
import os
from math import pi

# compas
from compas.colors import Color
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors

# jax_fdm
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import datastructure_update
from jax_fdm.equilibrium import fdm
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.visualization import Viewer

# ==========================================================================
# Parameters
# ==========================================================================

# NOTE: the input files are not committed to the repository. Generate them by
# running examples/pringle/pringle.py with record = True and export = True.
name = "pringle"

modify_view = True
show_grid = True
camera_zoom = -6  # number of zoom-out steps, negative to zoom out

interval = 50  # milliseconds between frames

animate = True
rotate_while_animate = False


# ==========================================================================
# Helper functions
# ==========================================================================

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

HERE = os.path.join(os.path.dirname(__file__), "../../data/json/")
FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_base.json"))

network0 = FDNetwork.from_json(FILE_IN)
model = EquilibriumModel(tmax=1, eta=1e-6)
structure = EquilibriumStructure.from_network(network0)
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
    viewer.renderer.camera.zoom(camera_zoom)  # number of steps, negative to zoom out
    viewer.renderer.camera.rotation.z = 2 * pi / 3  # rotation around the z axis

# draw network as plain geometry (keep a handle on the copy to animate it)
network_plain = network.copy(cls=Network)
viewer.add(network_plain,
           show_points=False,
           linewidth=5.0,
           linecolor=Color.grey().darkened())

# draw supports
support_objs = {}
for node in network.nodes_supports():
    x, y, z = network.node_coordinates(node)
    support_objs[node] = viewer.add(Point(x, y, z), pointcolor=Color.green(), pointsize=20)

# draw loads
loads = loads_draw(network)
load_objs = {}
for node, load in loads.items():
    load_objs[node] = viewer.add(load, linewidth=4.0, linecolor=Color.green().darkened())

# draw residual forces
residuals = residuals_draw(network)
residual_objs = {}
for node, residual in residuals.items():
    residual_objs[node] = viewer.add(residual, linewidth=4.0, linecolor=Color.pink())

# warm start model
params = recorder[0]
_ = model(params, structure)

# create update function
if animate:
    @viewer.on(interval=interval, frames=len(recorder))
    def wiggle(f):

        print(f"Current frame: {f + 1}/{len(recorder)}")
        params = recorder[f]
        eqstate = model(params, structure)

        # update network
        datastructure_update(network, eqstate, params)

        # sync the plain wireframe copy with the updated network
        for node in network.nodes():
            network_plain.node_attributes(node, "xyz", network.node_coordinates(node))

        # update supports
        for node, obj in support_objs.items():
            x, y, z = network.node_coordinates(node)
            obj.geometry.x = x
            obj.geometry.y = y
            obj.geometry.z = z

        # update loads and residual forces
        loads_update(loads, network)
        residuals_update(residuals, network)

        # re-read the mutated geometry into the render buffers
        for obj in viewer.scene.objects:
            obj.update(update_data=True)

        if rotate_while_animate:
            viewer.renderer.camera.rotate(dx=1, dy=0)

# show le crème
viewer.show()
