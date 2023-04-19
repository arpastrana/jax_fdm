from compas.geometry import Polyline
from compas.geometry import add_vectors

from jax_fdm.datastructures import FDNetwork


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
    visualize = True
    filepath = "arches.png"
    viz_options = {"show_loads": False,
                   "show_reactions": False,
                   "show_nodes": True,
                   "nodesize": 0.5}

    # instantiate a plotter (only for visualization, optional)
    if visualize:
        plotter = Plotter(figsize=(8, 5), dpi=200)

    # generate arches of increasing number of segments
    for num_segments in range(2, 50):

        # create network
        polyline = create_arch_polyline(arch_length, num_segments)
        network = create_arch_network(polyline, q_val * sqrt(num_segments), line_py)

        # create equiilibrium model from network
        model = EquilibriumModel(network)

        # extract fdm parameters from network
        q, xyz_fixed, loads = (jnp.asarray(p, dtype=jnp.float64) for p in network.parameters())

        # linear solve we are interested in timing
        xyz_free = model.nodes_free_positions(q, xyz_fixed, loads)

        # visualization (optional)
        if visualize:
            # run fdm (again) to get an FD network in static equilibrium
            network_eq = fdm(network)
            # add network in equilibrium to plotter
            plotter.add(network_eq, **viz_options)

    # save visualization plot
    if visualize:
        plotter.zoom_extents()
        plotter.save(filepath, dpi=300)
