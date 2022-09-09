# compas
from compas.geometry import Polyline
from compas.geometry import add_vectors

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm

# ==========================================================================
# Initial parameters
# ==========================================================================

arch_length = 5.0
num_segments = 5
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

from time import time
import autograd.numpy as np
from dfdm.equilibrium import EquilibriumModel

model = EquilibriumModel(network)
q = np.asarray(network.edges_forcedensities(), dtype=np.float64)

ntests = 10

times = 0.
for _ in range(ntests):
    start = time()
    eq_state = model(q)
    end = time() - start
    if _ == 0:
        print(end)
        continue
    times += end

print("autograd", times / ntests)

# ==========================================================================
# Run thee force density method
# ==========================================================================

import jax.numpy as jnp
from jax import jit, grad, jacobian
from jax_fdm.equilibrium import EquilibriumModel

model = EquilibriumModel(network)
q = jnp.asarray(network.edges_forcedensities(), dtype=jnp.float64)

times = 0.
for _ in range(ntests):
    start = time()
    eq_state = model(q).block_until_ready()
    end = time() - start
    if _ == 0:
        print(end)
        continue
    times += end

print("jax", times / ntests)
