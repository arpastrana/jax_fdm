# the essentials
import os

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
camera_zoom = 5  # number of zoom-out steps, negative to zoom out

interval = 30  # milliseconds between frames

animate = True
rotate_while_animate = True

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
viewer = Viewer(show_grid=show_grid)

# modify view
if modify_view:
    viewer.renderer.camera.zoom(camera_zoom)  # number of steps, negative to zoom out

# draw network, fused into batched mesh "soups" for fast per-frame buffer updates
network_obj = viewer.add(network, fuse=True, edgecolor="fd")

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

        # update mesh
        datastructure_update(network, eqstate, params)

        # update the render buffers of the force density scene object in place
        network_obj.update()

        if rotate_while_animate:
            viewer.renderer.camera.rotate(dx=1, dy=0)

# show le crème
viewer.show()
