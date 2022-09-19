import numpy as np

from jax_fdm.constraints import Constraint


class NodeConstraint(Constraint):
    """
    Base class for all constraints that pertain to a node in a network.
    """
    def __init__(self, key, bound_low, bound_up, **kwargs):
        super().__init__(bound_low=bound_low, bound_up=bound_up)
        self._key = key

    def index(self, model):
        """
        The index of the edge key in a structure.
        """
        return model.structure.node_index[self.key()]

    def key(self):
        """
        The key of the edge in the network.
        """
        return self._key


class NodeNormalAngleConstraint(Constraint):
    """
    Constraints the angle between the normal of the network at a node and a reference vector.
    """
    def __init__(self, key, vector, bound_low, bound_up):
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)
        self.vector_other = np.asarray(vector)

    def constraint(self, eqstate, model):
        """
        Returns the angle between the the node normal and the reference vector.
        """
        normal = self._node_normal(eqstate, model)
        return self._angle_vectors_numpy(normal, self.vector_other, deg=True)

    def _node_normal(self, eqstate, model):
        """
        Computes the vector normal at a node in a network.
        """
        index_node = self.index(model)
        index_others = model.structure.adjacency[index_node, :]
        xyz_node = eqstate.xyz[index_node, :]
        xyz_others = eqstate.xyz[index_others, :]
        vectors = xyz_node - xyz_others
        length_vectors = np.linalg.norm(vectors, axis=1)
        vectors = vectors / length_vectors
        return np.sum(vectors)

    @staticmethod
    def _angle_vectors_numpy(u, v, deg=False):
        """
        Compute the smallest angle between two vectors.
        """
        a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        a = max(min(a, 1), -1)
        if deg:
            return np.degrees(np.arccos(a))
        return np.arccos(a)
