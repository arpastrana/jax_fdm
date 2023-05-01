from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles

from compas.utilities import pairwise
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from functools import partial
import time

import pickle
import jax


# ==========================================================================
# Create the geometry of a saddle
# ==========================================================================

def diagonalize_mesh(mesh):
    """
    """
    dmesh = Mesh()

    # add vertices
    for vkey in mesh.vertices():
        dmesh.add_vertex(vkey)
        dmesh.vertex_attributes(vkey, "xyz", mesh.vertex_coordinates(vkey))

    # face centroids
    fkey_ckey = {}
    for fkey in mesh.faces():
        centroid = mesh.face_centroid(fkey)
        ckey = dmesh.add_vertex(attr_dict={k: v for k, v in zip("xyz", centroid)})
        fkey_ckey[fkey] = ckey

    # triangulate faces
    for u, v in mesh.edges():
        faces = [fkey for fkey in mesh.edge_faces(u, v) if fkey is not None]

        if len(faces) == 1:
            w = fkey_ckey[faces[0]]
            face = [u, v, w]
            dmesh.add_face(face)
        elif len(faces) == 2:
            t, w = (fkey_ckey[fkey] for fkey in faces)
            face = [u, t, v, w]
            dmesh.add_face(face)
        else:
            raise ValueError("The number of faces per edge is incorrect!")

    # unify mesh cycles
    mesh_unify_cycles(dmesh)

    return dmesh


def liftcorners_mesh(mesh, height):
    """
    """
    # Get indices of corner vertices
    boundary_vkeys = mesh.vertices_on_boundary()[:-1]
    corner_vkeys = [vkey for vkey in boundary_vkeys if len(mesh.vertex_neighbors(vkey)) == 3]
    corner_indices = [boundary_vkeys.index(vkey) for vkey in corner_vkeys]

    # Modify the z coordinate of the nodes on the boundary
    for i, (start, end) in enumerate(pairwise(corner_indices + [len(boundary_vkeys) - 1])):
        end = end + 1
        split_vkeys = boundary_vkeys[start:end]

        # if last polyedge, append first corner index to close the loop
        if i == len(corner_indices) - 1:
            split_vkeys.append(boundary_vkeys[0])

        # use a linear map between one corner vertex and the next one
        zs = remap_values(list(range(len(split_vkeys))), 0.0, height)

        # reverse vertex keys if i is zero or an odd number
        if i == 0 or i % 2 == 0:
            split_vkeys = split_vkeys[::-1]

        # modify z coordinate
        for vkey, z in zip(split_vkeys, zs):
            mesh.vertex_attribute(vkey, "z", z)

    return mesh


def create_saddle_geometry(length, num_segments, height):
    """
    """
    mesh = Mesh.from_meshgrid(dx=length, nx=num_segments)
    mesh = diagonalize_mesh(mesh)
    liftcorners_mesh(mesh, height)

    return mesh


def create_saddle_network(mesh, q):
    """
    """
    nodes, _ = mesh.to_vertices_and_faces()
    edges = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]

    network = FDNetwork.from_nodes_and_edges(nodes, edges)

    # assign supports
    for node in mesh.vertices_on_boundary():
        network.node_support(node)

    # assign edge force densities
    network.edges_forcedensities(q)

    return network


# ==========================================================================
# Main script
# ==========================================================================

if __name__ == "__main__":
    import jax.numpy as jnp

    from compas.geometry import Translation

    from jax_fdm.equilibrium import EquilibriumModel
    from jax_fdm.equilibrium import fdm
    from jax_fdm.visualization import Plotter
    from jax_fdm.visualization import Viewer

    # script parameters
    length_side = 10.
    height_corner = 5.
    q_val = 1.

    # viz controls
    visualize = False
    use_viewer = True
    filepath = "saddles.png"
    viz_options = {"show_loads": False,
                   "show_nodes": False,
                   }

    # instantiate a plotter (only for visualization, optional)
    if visualize:
        plotter = Plotter(figsize=(8, 5), dpi=200)

        if use_viewer:
            viewer = Viewer(width=1600, height=900, show_grid=False)

    num_reps = 5

    # generate saddles of increasing number of side segments
    info = []
    for i, num_segments in enumerate([4, 8, 16, 32, 64]):

        # create network
        dmesh = create_saddle_geometry(length_side, num_segments, height_corner)
        network = create_saddle_network(dmesh, q_val)

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
            T = Translation.from_vector([i * length_side * 1.2, 0., 0.0])
            network_eq = network_eq.transformed(T)
            plotter.add(network_eq, show_reactions=False, edgewidth=(0.2, 2.0), **viz_options)

            if use_viewer:
                if viz_options.get("edgewidth"):
                    del viz_options["edgewidth"]
                viewer.add(network_eq, show_reactions=True, edgewidth=(0.03, 0.3), **viz_options)

    pickle.dump(info, open("saddle_info.pkl", "wb"))

    # save visualization plot
    if visualize:
        if use_viewer:
            viewer.show()

        plotter.zoom_extents()
        plotter.save(filepath, dpi=300)
        plotter.show()
