import numpy as np

import jax.numpy as jnp

from jax import vmap

from jax_fdm.geometry import angle_vectors
from jax_fdm.geometry import normalize_vector

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.node import NodeGoal


class NodeNormalAngleGoal(ScalarGoal, NodeGoal):
    """
    Reach a target value for the angle formed by the node normal and a reference vector.
    """
    def __init__(self, key, vector, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)
        self._vector = None
        self.vector = vector

        self.index_faces = None
        self.mask_faces = None

    @property
    def vector(self):
        """
        The vector to take the angle with.
        """
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def vectors(self):
        """
        Create a matrix of vectors.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return matrix

    def init(self, model):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model)
        self.vector = self.vectors()
        self.index_faces, self.mask_faces = self.faces_indices(model)

    def faces_indices(self, model):
        """
        Get the node indices of the faces connected to a node.

        TODO: Please refactor me! I am illegible :(
        """
        index_max = max(self.index) + 1

        num_nodes_max = -1
        num_faces_max = -1
        for idx in self.index:
            num_faces = len(self.node_faces(model, idx))
            if num_faces > num_faces_max:
                num_faces_max = num_faces
            num_nodes = len(max(self.node_faces_indices(model, idx), key=lambda x: len(x)))
            if num_nodes > num_nodes_max:
                num_nodes_max = num_nodes

        index_faces = np.zeros((index_max, num_faces_max, num_nodes_max + 1))
        mask_faces = np.zeros((index_max, num_faces_max, num_nodes_max + 1))

        for idx in self.index:
            nfn = self.node_faces_indices(model, idx)
            for i, face_nodes in enumerate(nfn):
                index_faces[idx, i, :len(face_nodes)] = face_nodes
                mask_faces[idx, i, :len(face_nodes)] = [1] * len(face_nodes)

                index_faces[idx, i, -1] = face_nodes[-1]
                mask_faces[idx, i, -1] = 1

        index_faces = jnp.asarray(index_faces, dtype=jnp.int64)
        mask_faces[mask_faces == 0] = jnp.nan
        mask_faces = jnp.asarray(mask_faces, dtype=jnp.float64)

        return index_faces, mask_faces

    @staticmethod
    def node_faces(model, index):
        """
        Return an iterable with the indices of the faces connected to a node.
        """
        connectivity = model.structure.connectivity_faces
        return np.flatnonzero(connectivity[:, index])

    def node_faces_indices(self, model, index):
        """
        Return an iterable with the indices of the nodes of the faces connected to a node.
        """
        fidx = self.node_faces(model, index)
        return [model.structure.face_node_index[idx] for idx in fidx]

    @staticmethod
    def nan_normal_polygon(polygon):
        """
        Compute the normal of a polygon that contains nan entries.
        """
        centroid = jnp.nanmean(polygon, axis=0)
        op = polygon - centroid
        op_shifted = jnp.roll(op, 1, axis=0)
        ns = 0.5 * jnp.cross(op_shifted, op)

        return jnp.nansum(ns, axis=0)

    def face_normal(self, eqstate, face, mask):
        """
        Calculate the normal of a face.
        """
        polygon = eqstate.xyz[face, :]
        mask = jnp.reshape(mask, (-1, 1))
        polygon = mask * polygon

        return self.nan_normal_polygon(polygon)

    def node_normal(self, eqstate, index):
        """
        Get the unitized normal vector at a node.
        """
        faces = self.index_faces[index]
        masks = self.mask_faces[index]
        normals = vmap(self.face_normal, in_axes=(None, 0, 0))(eqstate, faces, masks)

        return normalize_vector(jnp.nanmean(normals, axis=0))

    def prediction(self, eqstate, index):
        """
        Returns the angle between the node normal and the reference vector.
        """
        normal = self.node_normal(eqstate, index)

        return angle_vectors(normal, self.vector[index, :])
