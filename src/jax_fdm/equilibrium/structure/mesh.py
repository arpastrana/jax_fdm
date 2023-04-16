import numpy as np

from compas.numerical import adjacency_matrix
from compas.numerical import connectivity_matrix
from compas.numerical import face_matrix


# ==========================================================================
# Structure
# ==========================================================================

class EquilibriumStructureMesh:
    """
    An equilibrium structure.
    """
    def __init__(self, mesh):
        self._datastruct = mesh
        self.network = mesh

        self._connectivity = None
        self._connectivity_free = None
        self._connectivity_fixed = None
        self._connectivity_faces = None

        self._connectivity_edges_faces = None

        self._edges = None
        self._nodes = None
        self._faces = None

        self._edges_faces = None

        self._free_nodes = None
        self._fixed_nodes = None
        self._freefixed_nodes = None

        self._node_index = None
        self._edge_index = None
        self._face_index = None

    @property
    def datastruct(self):
        """
        A COMPAS datastructure.
        """
        return self._datastruct

# ==========================================================================
# Essential collections
# ==========================================================================

    @property
    def edges(self):
        """
        A list with the edge keys of the structure.
        """
        if not self._edges:
            self._edges = list(self.datastruct.edges())
        return self._edges

    @property
    def nodes(self):
        """
        A list with the node keys of the structure.
        """
        if not self.nodes:
            self._nodes = list(self.datastruct.vertices())

    @property
    def faces(self):
        """
        A list with the face keys of the structure.
        """
        if not self._faces:
            self._faces = [tuple(self.datastruct.face_vertices(fkey)) for fkey in self.datastruct.faces()]
        return self._faces

# ==========================================================================
# Convenience collections
# ==========================================================================

    @property
    def free_nodes(self):
        """
        Returns a list with the indices of the anchored nodes.
        """
        if not self._free_nodes:
            self._free_nodes = [self.node_index[node] for node in self.datastruct.vertices_free()]
        return self._free_nodes

    @property
    def fixed_nodes(self):
        """
        Returns a list with the indices of the anchored nodes.
        """
        if not self._fixed_nodes:
            self._fixed_nodes = [self.node_index[node] for node in self.datastruct.vertices_fixed()]
        return self._fixed_nodes

    @property
    def freefixed_nodes(self):
        """
        A list with the node keys of all the nodes sorted by their node index.
        TODO: this method must be refactored to be more transparent.
        """
        if not self._freefixed_nodes:
            freefixed_nodes = self.free_nodes + self._fixed_nodes
            indices = {node: index for index, node in enumerate(freefixed_nodes)}
            sorted_indices = []
            for _, index in sorted(indices.items(), key=lambda item: item[0]):
                sorted_indices.append(index)
            self._freefixed_nodes = tuple(sorted_indices)

        return self._freefixed_nodes

# ==========================================================================
# Indexing mappings
# ==========================================================================

    @property
    def node_index(self):
        """
        A dictionary between node keys and their enumeration indices.
        """
        if not self._node_index:
            self._node_index = self.datastruct.vertex_index()
        return self._node_index

    @property
    def edge_index(self):
        """
        A dictionary between edge keys and their enumeration indices.
        """
        if not self._edge_index:
            self._edge_index = self.datastruct.uv_index()
        return self._edge_index

    @property
    def face_index(self):
        """
        A dictionary between face keys and their enumeration indices.
        """
        if not self._face_index:
            self._face_index = {face: idx for idx, face in enumerate(self.faces)}
        return self._face_index

# ==========================================================================
# Mappings between collections
# ==========================================================================

    @property
    def edges_faces(self):
        """
        The connectivity matrix of the edges and the faces of a mesh.
        """
        if self._edges_faces is None:

            datastruct = self.datastruct
            face_index = self.face_index
            node_index = self.node_index

            edges_faces = []
            for u, v in self.edges:

                findices = []
                for fkey in datastruct.edge_faces(u, v):

                    if fkey is None:
                        continue

                    face = [node_index[vkey] for vkey in datastruct.face_vertices(fkey)]
                    findex = face_index[tuple(face)]
                    findices.append(findex)

                edges_faces.append(tuple(findices))

            self._edges_faces = tuple(edges_faces)

        return self._edges_faces

# ==========================================================================
# Matrices
# ==========================================================================

    @property
    def connectivity(self):
        """
        The connectivity matrix of the edges of a datastrcture.
        """
        if self._connectivity is None:
            node_idx = self.node_index
            edges = [(node_idx[u], node_idx[v]) for u, v in self.edges]
            self._connectivity = connectivity_matrix(edges, "array")
        return self._connectivity

    @property
    def connectivity_faces(self):
        """
        The connectivity matrix of the nodes and the faces of a mesh.
        """
        if self._connectivity_faces is None:
            node_idx = self.node_index
            face_nodes = [[node_idx[node] for node in face] for face in self.faces]
            self._connectivity_faces = face_matrix(face_nodes, "array", normalize=True)
        return self._connectivity_faces

    @property
    def connectivity_edges_faces(self):
        """
        The connectivity matrix between edges and faces of a mesh.
        """
        if self._connectivity_edges_faces is None:
            connectivity = np.zeros((len(self.edges), len(self.faces)))
            edges_faces = self.edges_faces
            for eindex, findex in enumerate(self.edges_faces):
                connectivity[eindex, findex] = 1.
            self._connectivity_edges_faces = connectivity
        return self._connectivity_edges_faces


    @property
    def connectivity_fixed(self):
        """
        The connectivity of the fixed nodes of the network.
        """
        if self._connectivity_fixed is None:
            fixed_nodes = self.fixed_nodes
            self._connectivity_fixed = self.connectivity[:, fixed_nodes]
        return self._connectivity_fixed

    @property
    def connectivity_free(self):
        """
        The connectivity of the free nodes of the network.
        """
        if self._connectivity_free is None:
            free_nodes = self.free_nodes
            self._connectivity_free = self.connectivity[:, free_nodes]
        return self._connectivity_free
