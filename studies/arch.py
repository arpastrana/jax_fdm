from compas.geometry import Polyline
from compas.geometry import add_vectors

import jax
from jax_fdm.datastructures import FDNetwork

from functools import partial
import time

import pickle

# ==========================================================================
# Create the geometry of an arch
# ==========================================================================

def create_arch_polyline(arch_length, num_segments):
    """
    """
    start = [-arch_length / 2.0, 0.0, 0.0]
    end = add_vectors(start, [arch_length, 0.0, 0.0])
    curve = Polyline([start, end])
    points = curve.divide_polyline(num_segments)

    return Polyline(points)


def create_arch_network(polyline, q, line_py):
    """
    """
    # instantiate network
    lines = polyline.lines
    network = FDNetwork.from_lines(lines)

    # assign supports
    network.node_support(key=0)
    network.node_support(key=len(lines))

    # set initial point loads to all nodes of the network
    py = line_py * polyline.length / len(polyline.points)
    network.nodes_loads([0.0, py, 0.0])

    # set edge force densities
    network.edges_forcedensities(q)

    return network


# ==========================================================================
# Main script
# ==========================================================================

if __name__ == "__main__":
    # imports
    from math import sqrt
    import jax.numpy as jnp
    from jax_fdm.equilibrium import EquilibriumModel
    from jax_fdm.equilibrium import fdm
    from jax_fdm.visualization import Plotter

    # script parameters
    arch_length = 10.0
    line_py = -1.0
    q_val = 1.0

    # viz controls
    visualize = False
    filepath = "arches.png"
    viz_options = {"show_loads": False,
                   "show_reactions": False,
                   "show_nodes": True,
                   "nodesize": 0.5}

    # instantiate a plotter (only for visualization, optional)
    if visualize:
        plotter = Plotter(figsize=(8, 5), dpi=200)

    # generate arches of increasing number of segments
    num_segments = 10
    num_reps = 5
    multiple = 2.0

    info = []
    for i in range(10):
        # create network
        polyline = create_arch_polyline(arch_length, num_segments)
        network = create_arch_network(polyline, q_val * sqrt(num_segments), line_py)

        # create equiilibrium model from network
        model = EquilibriumModel(network)

        # extract fdm parameters from network
        q, xyz_fixed, loads = (jnp.asarray(p, dtype=jnp.float64) for p in network.parameters())

        # linear solve we are interested in timing
        sparse_fn = jax.jit(partial(model.nodes_free_positions, sparsesolve=True))
        no_sparse_fn = jax.jit(partial(model.nodes_free_positions, sparsesolve=False))
        # JIT the functions first

        jit_start = time.time()
        sparse_fn(q, xyz_fixed, loads)
        jit_end = time.time()
        sparse_jit_time = jit_end - jit_start

        jit_start = time.time()
        no_sparse_fn(q, xyz_fixed, loads)
        jit_end = time.time()
        no_sparse_jit_time = jit_end - jit_start

        sparse_times = []
        for j in range(num_reps):
            start = time.time()
            xyz_free = sparse_fn(q, xyz_fixed, loads)
            end = time.time()
            sparse_times.append(end - start)

        no_sparse_times = []
        for j in range(num_reps):
            dense_start = time.time()
            xyz_free = no_sparse_fn(q, xyz_fixed, loads)
            dense_end = time.time()
            no_sparse_times.append(dense_end - dense_start)

        info.append({"num_segments": num_segments,
                     "sparse_jit_time": sparse_jit_time,
                     "no_sparse_jit_time": no_sparse_jit_time,
                     "sparse_times": sparse_times,
                     "no_sparse_times": no_sparse_times})

        print(f"number of segments: {num_segments} "
              f"sparse mean time: {sum(sparse_times) / num_reps} "
              f"dense mean time: {sum(no_sparse_times) / num_reps} "
              f"sparse jit: {sparse_jit_time} "
              f"dense jit: {no_sparse_jit_time}")

        # visualization (optional)
        if visualize:
            # run fdm (again) to get an FD network in static equilibrium
            network_eq = fdm(network)
            # add network in equilibrium to plotter
            plotter.add(network_eq, **viz_options)

        num_segments = int(num_segments * multiple)

    pickle.dump(info, open("arches_info.pkl", "wb"))

    # save visualization plot
    if visualize:
        plotter.zoom_extents()
        plotter.save(filepath, dpi=300)
