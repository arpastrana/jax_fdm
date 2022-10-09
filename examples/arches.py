import jax.numpy as jnp

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Translation
from compas.geometry import add_vectors

# jax_fdm
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.goals import NodeResidualDirectionGoal
from jax_fdm.losses import SquaredError
from jax_fdm.losses import Loss
from jax_fdm.optimization import SLSQP
from jax_fdm.visualization import Viewer

# ==========================================================================
# Initial parameters
# ==========================================================================

length_arch = 5.0
num_segments = 10
q_init = -1
pz = -0.3
start = [0.0, 0.0, 0.0]

# ==========================================================================
# Create the base geometry of the arch
# ==========================================================================

points = []
length_segment = length_arch / num_segments
for i in range(num_segments + 1):
    point = add_vectors(start, [i * length_segment, 0.0, 0.0])
    points.append(point)

# ==========================================================================
# Create arch network
# ==========================================================================

network = FDNetwork()

for idx, point in enumerate(points):
    x, y, z = point
    network.add_node(idx, x=x, y=y, z=z)

for u, v in zip(range(0, num_segments), range(1, num_segments + 1)):
    network.add_edge(u, v)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
network.node_support(0)
network.node_support(num_segments)

# apply loads to unsupported nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all nodes
for edge in network.edges():
    network.edge_forcedensity(edge, q_init)

# ==========================================================================
# Instantiate viewer
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

# reference arch
viewer.add(network, as_wireframe=True, show_points=False, linewidth=4.0)

# color map
cmap = ColorMap.from_mpl("viridis")

# ==========================================================================
# Define goals
# ==========================================================================

constrained_networks = []
vertical_comps = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

for idx, vertical_comp in enumerate(vertical_comps):

    goals = []
    goals.append(NodeResidualDirectionGoal(0, target=[-1.0, 0.0, -vertical_comp]))
    goals.append(NodeResidualDirectionGoal(num_segments, target=[1.0, 0.0, -vertical_comp]))

# ==========================================================================
# Optimization
# ==========================================================================

    constrained_network = constrained_fdm(network,
                                          optimizer=SLSQP(),
                                          loss=Loss(SquaredError(goals)),
                                          bounds=(-jnp.inf, 0.0),
                                          maxiter=200,
                                          tol=1e-9)

    constrained_networks.append(constrained_networks)

# ==========================================================================
# Visualization
# ==========================================================================

    # equilibrated arch
    color = cmap(1.0 - idx / len(vertical_comps))

    t_vector = [0.0, -idx, 0.0]
    T = Translation.from_vector([0.0, -idx, 0.0])

    # reference arch
    viewer.add(network.transformed(T),
               as_wireframe=True,
               show_points=False,
               linewidth=2.0,
               color=Color.grey().darkened())

    # constrained arch
    c_network = constrained_network.transformed(T)
    viewer.add(c_network,
               edgewidth=(0.01, 0.15),
               edgecolor=color,
               reactionscale=0.25,
               reactioncolor=Color.pink())

# show le crème
viewer.show()
