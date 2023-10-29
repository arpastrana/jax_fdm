import jax.numpy as jnp

from jax import vmap

from jax_fdm.geometry import area_triangle
from jax_fdm.geometry import normal_polygon
from jax_fdm.geometry import length_vector
from jax_fdm.geometry import line_lcs
from jax_fdm.geometry import polygon_lcs


# ==========================================================================
# Face loads
# ==========================================================================

from jax_fdm.geometry import normalize_vector


def nodes_load_from_faces(xyz, faces_load, structure, is_local=False):
    """
    Calculate the tributary face loads aplied to the nodes of a structure.
    """
    faces = structure.faces_indexed
    faces_load = calculate_faces_load(xyz, faces, faces_load, is_local)
    edges_load = edges_tributary_faces_load(xyz, faces_load, structure)
    nodes_load = nodes_tributary_edges_load(edges_load, structure)

    return nodes_load


def calculate_faces_load(xyz, faces, faces_load, is_local):
    """
    Transform the face loads to the XYZ cartesian coordinate system if needed.
    """
    if not is_local:
        return faces_load

    faces_load_lcs = vmap(face_load_lcs, in_axes=(None, 0, 0))
    loads = faces_load_lcs(xyz, faces, faces_load)

    return loads


def face_load_lcs(xyz, face, face_load):
    """
    Transform the load vector applied to the face to a vector in its local coordinate system.
    """
    def face_xyz(xyz, face):
        """
        Get this face XYZ coordinates from XYZ vertices array.
        """
        face = jnp.ravel(face)

        xyz_face = xyz[face, :]
        xyz_repl = xyz_face[0, :]

        # NOTE: Replace -1 with first entry to avoid nans in gradient computation
        # This was a pesky bug, since using nans as replacement did not cause
        # issues with the forward computation of normals, but it does for
        # the backward pass.
        xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

        return xyz_face

    fxyz = face_xyz(xyz, face)

    normal = normal_polygon(fxyz)
    is_zero_normal = jnp.allclose(normal, 0.0)
    lcs = jnp.where(is_zero_normal, jnp.eye(3), polygon_lcs(fxyz))

    load = face_load @ lcs

    return load


def edges_tributary_faces_load(xyz, faces_load, structure):
    """
    Calculate the face area load taken by every edge in a datastructure.
    """
    c_vertices = structure.connectivity
    c_faces = structure.connectivity_edges_faces

    face_centroids = structure.connectivity_faces_vertices @ xyz
    loads_fn = vmap(edge_tributary_faces_load, in_axes=(0, 0, None, None, None))

    return loads_fn(c_vertices, c_faces, xyz, faces_load, face_centroids)


def edge_tributary_faces_load(c_edge_nodes, c_edge_faces, xyz, faces_load, face_centroids):
    """
    Calculate the face area load taken by one edge in a datastructure.
    """
    indices = jnp.flatnonzero(c_edge_nodes, size=2)
    line = xyz[indices, :]

    findices = jnp.flatnonzero(c_edge_faces, size=2, fill_value=-1)
    floads = faces_load[findices, :]

    centroids = face_centroids[findices, :]

    # NOTE: correct loads if negative face indices exist (e.g. faces on boundary)
    areas = vmap(edge_tributary_face_area, in_axes=(None, 0))(line, centroids)
    areas = jnp.where(findices >= 0, areas.ravel(), 0.0)

    return areas @ floads


def edge_tributary_face_area(line, centroid):
    """
    The triangle-based, face tributary area of an edge.
    """
    triangle = jnp.vstack((line, jnp.reshape(centroid, (1, 3))))

    return area_triangle(triangle)


def _faces_load_2(xyz, faces, faces_load, is_local):
    """
    Transform the face loads to the XYZ cartesian coordinate system.
    """
    def _faces_load(xyz, face, face_load, is_local):
        return jnp.where(is_local, face_load_lcs(xyz, face, face_load), face_load)

    floads, is_local = faces_load
    vmap_facesload = vmap(_faces_load, in_axes=(None, 0, 0, 0))

    return vmap_facesload(xyz, faces, faces_load, is_local)


# ==========================================================================
# Edge loads
# ==========================================================================


def nodes_load_from_edges(xyz, edges_load, structure, is_local=False):
    """
    Calculate the tributary edge loads aplied to the nodes of a structure.
    """
    edges = structure.edges_indexed
    edges_load = calculate_edges_load(xyz, edges, edges_load, is_local)
    edges_load = edges_tributary_edges_load(xyz, edges_load, structure)

    return nodes_tributary_edges_load(edges_load, structure)


def calculate_edges_load(xyz, edges, edges_load, is_local):
    """
    Transform the edges load to the XYZ cartesian coordinate system if needed.
    """
    if not is_local:
        return edges_load

    edges_load_lcs = vmap(edge_load_lcs, in_axes=(None, 0, 0))

    return edges_load_lcs(xyz, edges, edges_load)


def edge_load_lcs(xyz, edge, edge_load):
    """
    Transform the load vector applied to an edge to a vector in its local coordinate system.
    """
    def edge_xyz(xyz, edge):
        return xyz[edge, :]

    exyz = edge_xyz(xyz, edge)
    lcs = line_lcs(exyz)

    return edge_load @ lcs


def edges_tributary_edges_load(xyz, edges_load, structure):
    """
    Calculate the face area load taken by every edge in a datastructure.
    """
    # TODO: edge lengths calculated by the FDM equilibrium model, inject?
    edges_vector = structure.connectivity @ xyz
    edges_length = length_vector(edges_vector)

    # scale edge load by edge length
    return edges_load * edges_length


# ==========================================================================
# Node helpers
# ==========================================================================

def nodes_tributary_edges_load(edges_load, structure):
    """
    Calculate the load vector applied to the nodes based on the edge loads.
    """
    return 0.5 * jnp.abs(structure.connectivity).T @ edges_load


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    import os
    from jax_fdm import DATA
    from jax_fdm.datastructures import FDMesh
    from jax_fdm.equilibrium import EquilibriumMeshStructure
    from jax_fdm.equilibrium import EquilibriumMeshStructureSparse

    # define tests
    def test_faces_load(mesh, pz, dtol=1e-1, sparse=False, is_local=False, verbose=False):

        mesh.faces_loads([0.0, 0.0, pz])

        if sparse:
            structure = EquilibriumMeshStructureSparse.from_mesh(mesh)
        else:
            structure = EquilibriumMeshStructure.from_mesh(mesh)

        xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        faces_load = jnp.asarray(mesh.faces_loads())

        faces = structure.faces_indexed
        nodes_load = nodes_load_from_faces(xyz, faces_load, structure)

        faces_load = calculate_faces_load(xyz, faces, faces_load, is_local)
        edges_load = edges_tributary_faces_load(xyz, faces_load, structure)

        sum_nodes_load = jnp.sum(nodes_load, axis=0)
        sum_edges_load = jnp.sum(edges_load, axis=0)

        compas_load = jnp.zeros(3)
        compas_faces_loads = []
        compas_faces_areas = []
        for face in mesh.faces():
            face_area = mesh.face_area(face)
            face_load = jnp.array(mesh.face_load(face)) * face_area
            compas_faces_areas.append(face_area)
            if is_local:
                fxyz = jnp.array(mesh.face_coordinates(face))
                face_load = face_load @ polygon_lcs(fxyz)
            compas_faces_loads.append(face_load)
            compas_load = compas_load + face_load

        if verbose:
            print(f"Face test: {sum_nodes_load=}, {sum_edges_load=}, {compas_load=:}")
            print("num faces", mesh.number_of_faces())
            print("num edges", mesh.number_of_edges())
            print("num nodes", mesh.number_of_vertices())
            print("edges load shape", edges_load.shape)
            print("nodes load shape", nodes_load.shape)
            print("faces load shape", faces_load.shape)

        # Test if resultant mesh loads match
        assert jnp.allclose(sum_edges_load, compas_load)
        assert jnp.allclose(sum_nodes_load, compas_load)

        # print(f"{compas_faces_loads=}")
        # Compare node loads individually
        compas_nodes_load = []
        face_index = {face: idx for idx, face in enumerate(mesh.faces())}

        if is_local:
            for vertex in structure.vertices:

                vertex = int(vertex)
                vertex_faceloads = [compas_faces_loads[face_index[face]] for face in mesh.vertex_faces(vertex)]
                vertex_faceareas = [compas_faces_areas[face_index[face]] for face in mesh.vertex_faces(vertex)]

                vertex_faceloads_unit = jnp.array(vertex_faceloads) / jnp.reshape(jnp.array(vertex_faceareas), (-1, 1))
                vload = jnp.sum(vertex_faceloads_unit, axis=0)
                vload = normalize_vector(vload) * pz

                # vload = jnp.sum(jnp.array(vertex_faceloads), axis=0)
                # varea = jnp.sum(jnp.array(vertex_faceareas), axis=0)
                # vload = vload / varea
                # compas_vertex_area = mesh.vertex_area(vertex)
                vload_unit = normalize_vector(vload)
                vnormal = jnp.array(mesh.vertex_normal(vertex))

                distance = jnp.linalg.norm(vload_unit - vnormal)

                # NOTE: Only checking if the load vector is in the same direction as the vertex normal
                # This is not checking that the load magnitudes are equal
                assert distance <= dtol, f"Distance: {distance}. Not equal: JAX: {vload_unit} vs. COMPAS: {vnormal}"
                compas_nodes_load.append(vload)

                # vload_unit = normalize_vector(jnp.sum(jnp.array(vertex_faceloads), axis=0))
                # vload = vload_unit * pz
                # compas_nodes_load.append(vload)

        else:
            compas_nodes_load = []
            for vertex in structure.vertices:
                vertex = int(vertex)
                vload = mesh.vertex_area(vertex) * pz
                compas_nodes_load.append([0.0, 0.0, vload])

            compas_nodes_load = jnp.array(compas_nodes_load)
            assert jnp.allclose(nodes_load, compas_nodes_load), f"Not equal: JAX:\n{nodes_load} vs. COMPAS:\n{compas_nodes_load}"

        mesh.faces_loads([0.0, 0.0, 0.0])
        # exit

    def test_edges_load(mesh, pz, dtol=1e-1, sparse=False, is_local=False, verbose=False):
        mesh.edges_loads([0.0, 0.0, pz])

        if sparse:
            structure = EquilibriumMeshStructureSparse.from_mesh(mesh)
        else:
            structure = EquilibriumMeshStructure.from_mesh(mesh)

        xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])

        edges = structure.edges_indexed
        edges_load = jnp.asarray(mesh.edges_loads())

        nodes_load = nodes_load_from_edges(xyz, edges_load, structure, is_local)

        edges_load_eff = calculate_edges_load(xyz, edges, edges_load, is_local)
        edges_load_trib = edges_tributary_edges_load(xyz, edges_load_eff, structure)

        sum_nodes_load = jnp.sum(nodes_load, axis=0)
        sum_edges_load = jnp.sum(edges_load_trib, axis=0)

        compas_load = jnp.zeros(3)
        for u, v in mesh.edges():
            edge_length = mesh.edge_length(u, v)
            edge_load = jnp.array(mesh.edge_load((u, v))) * edge_length
            if is_local:
                exyz = jnp.array(mesh.edge_coordinates(u, v))
                edge_load = edge_load @ line_lcs(exyz)
            compas_load = compas_load + edge_load

        if verbose:
            print(f"Edge test: {sum_nodes_load=}, {sum_edges_load=}, {compas_load=}")
            print("num edges", mesh.number_of_edges())
            print("num nodes", mesh.number_of_vertices())
            print("edges load shape", edges_load.shape)
            print("nodes load shape", nodes_load.shape)

        # Test if resultant mesh loads match
        assert jnp.allclose(sum_edges_load, compas_load)
        assert jnp.allclose(sum_nodes_load, compas_load)

    def test_mesh_loads(mesh, pz, dtol, sparse, is_local, verbose):
        test_faces_load(mesh, pz, dtol=dtol, sparse=sparse, is_local=is_local)
        test_edges_load(mesh, pz, dtol=dtol, sparse=sparse, is_local=is_local, verbose=verbose)

    # run tests
    pz = 2.0
    sparse = True
    is_local = False
    dtol = 1e-1
    verbose = False

    print("Test mesh grid")
    # for nx in range(1, 2):
    for nx in range(5, 10):
        mesh = FDMesh.from_meshgrid(dx=2, nx=nx)
        test_mesh_loads(mesh, pz, dtol=dtol, sparse=sparse, is_local=is_local, verbose=verbose)
    print()

    print("Test mesh polyhedron")
    for f in (4, 6, 8, 12, 20):
        mesh = FDMesh.from_polyhedron(f)
        test_mesh_loads(mesh, pz, dtol=dtol, sparse=sparse, is_local=is_local, verbose=verbose)
    print()

    print("Test mesh pillow")
    for name in ("uncstr_opt", "cstr_opt", "free", "loaded"):
        filepath = os.path.join(DATA, f"json/pillow_{name}_mesh.json")
        mesh = FDMesh.from_json(filepath)
        test_mesh_loads(mesh, pz, dtol=dtol, sparse=sparse, is_local=is_local, verbose=verbose)

    print("\nVery good! Slow down, cowboy!")
