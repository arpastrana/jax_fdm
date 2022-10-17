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
from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import RootMeanSquaredError
from jax_fdm.losses import Loss

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import jax.numpy as jnp


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

q0 = -0.2
dq = 1.2
px, py, pz = 0.0, 0.0, -0.2  # loads at each node

# ==========================================================================
# Read in optimization history
# ==========================================================================

HERE = os.path.join(os.path.dirname(__file__), "../../data/json/")
FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_history.json"))
recorder = OptimizationRecorder.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_base.json"))
network = FDNetwork.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_target.json"))
network_target = FDNetwork.from_json(FILE_IN)


# data
supports = [node for node in network.nodes() if network.is_leaf(node)]
# network.nodes_supports(supports)
# network.nodes_loads([px, py, pz], keys=network.nodes_free())
# network.edges_forcedensities(q=q0)
model = EquilibriumModel(network)

# viewer = Viewer(width=1600, height=900, show_grid=False)
# viewer.add(network, as_wireframe=True)
# viewer.add(network_target, as_wireframe=True)

# viewer.show()

# from compas.geometry import distance_point_point

goals = []
for node in network.nodes():
    if node in supports:
        continue

    xyz = network_target.node_coordinates(node)
    # print(distance_point_point(network.node_coordinates(node), xyz))
    goal = NodePointGoal(node, xyz)
    goals.append(goal)

# raise

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error = RootMeanSquaredError(goals, alpha=1.0)
loss = Loss(squared_error)

# build goal collections
from jax_fdm.optimization import SLSQP
optimizer = SLSQP()
for term in loss.terms_error:

    goal_collections = optimizer.collect_goals(term.goals)
    for goal_collection in goal_collections:
        goal_collection.init(model)
    term.collections = goal_collections

# loss_value = loss(jnp.array(network.edges_forcedensities()), model)
# print(loss_value)
# ==========================================================================
# Visualization
# ==========================================================================

# x1 = np.arange(0, -0.2, -0.002)
# y1 = np.arange(0, -0.2, -0.002)
# x2 = np.arange(3.9, 3.7, -0.002)
# y2 = np.arange(0, 1, 0.01)
# x3 = np.arange(0, 1.8, 0.018)
# y3 = np.array(x3**2)



# for q in recorder.history():
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
# fig = plt.figure(figsize=(8, 6))

ax.set_xlabel("Optimization iterations")
ax.set_ylabel("Loss value")
ax.set_yscale("log")
ax.grid(which="major")

print(len(recorder.history))

qs = recorder.history
losses = [loss(np.array(q), model) for q in qs]
x = np.arange(len(qs))

# print(x)
# plt.plot(x, losses)
# plt.show()
# raise

# print(losses)

# line = ax.plot(losses[1:i], losses[1:i], color = 'blue', lw=1)

def animate(i):
    ax.clear()
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 150)
    # ax.grid()

    # ax.set_yscale("log")
    ax.set_xlabel("Optimization iterations")
    ax.set_ylabel("Loss value")
    ax.grid(which="major")

    line, = ax.plot(x[0:i], losses[0:i], lw=2, color="tab:orange")
    point, = ax.plot(x[i], losses[i], marker='.', color='tab:orange')
    return line, point

ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=len(qs))
ani.save("temp/loss_func.gif", dpi=150, writer=PillowWriter(fps=8))
print("saved")
plt.show()
