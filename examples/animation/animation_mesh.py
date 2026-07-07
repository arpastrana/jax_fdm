# the essentials
import os
from math import pi

# jax_fdm
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import datastructure_update
from jax_fdm.equilibrium import fdm
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.visualization import Viewer

# ==========================================================================
# Parameters
# ==========================================================================

# NOTE: the input files are not committed to the repository. Generate them by
# running examples/monkey_saddle/monkey_saddle.py with record = True and export = True.
name = "monkey_saddle"

modify_view = True
show_grid = True
camera_zoom = -10  # number of zoom-out steps, negative to zoom out

interval = 50  # milliseconds between frames

animate = True
rotate_while_animate = False

# ==========================================================================
# Read in force density mesh
# ==========================================================================

HERE = os.path.join(os.path.dirname(__file__), "../../data/json/")
FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_base.json"))

mesh0 = FDMesh.from_json(FILE_IN)
model = EquilibriumModel(tmax=1, eta=1e-6)
structure = EquilibriumMeshStructure.from_mesh(mesh0)
mesh = fdm(mesh0)

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

# draw mesh
viewer.add(mesh,
           edgewidth=(0.05, 0.25),
           edgecolor="fd",
           show_nodes=False,
           nodesize=0.5,
           show_reactions=True,
           show_loads=True
           )

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
        datastructure_update(mesh, eqstate, params)

        # update the scene objects drawn by the force density artists
        for artist in viewer.artists:
            artist.update()

        if rotate_while_animate:
            viewer.renderer.camera.rotate(dx=1, dy=0)

# show le crème
viewer.show()
