# the essentials
import os
from math import pi
import numpy as np

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

name = "monkey_saddle"

modify_view = True
show_grid = True
camera_zoom = -35  # -35 for monkey saddle, 0 for pringle, 14 for dome, -70 for butt

decimate = False
decimate_step = 0

interval = 50  # 50
timeout = None
fps = 8

animate = True
rotate_while_animate = True

save = True

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
    # number of steps, negative to zoom out
    viewer.view.camera.zoom(camera_zoom)
    # set rotation around z axis to zero
    viewer.view.camera.rotation[2] = 2 * pi / 3
    # set rotation around z axis to zero
    viewer.view.camera.rotation_delta = (2 / 3) * pi / len(recorder.history)

# draw network
viewer.add(network,
           edgewidth=(0.05, 0.25),
           edgecolor="fd",
           show_nodes=False,
           nodesize=0.5,
           show_reactions=True,
           show_loads=True
           )

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
                      "record_path": f"temp/{name}_{fps}fps_viewer.gif"}

    @viewer.on(**config_animate)
    def wiggle(f):

        print(f"Current frame: {f + 1}/{len(recorder.history)}")
        q = np.array(recorder.history[f])
        eqstate = model(q)

        # update network
        network_update(network, eqstate)

        # update all buffer objects in the view
        for artist in viewer.artists:
            artist.update()
            for obj in artist.objects:
                obj.update()

        if rotate_while_animate:
            viewer.view.camera.rotate(dx=1, dy=0)

# show le crème
viewer.show()
