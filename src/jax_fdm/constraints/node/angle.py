import jax.numpy as jnp

from jax_fdm.geometry import angle_vectors
from jax_fdm.geometry import normal_polygon

from jax_fdm.constraints.node import NodeConstraint


class NodeAngleConstraint(NodeConstraint):
    """
    Constraints the angle between the normal of the network at a node and a reference vector.
    """
    def __init__(self, key, vector, bound_low, bound_up):
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)

        self._vector = None
        self.vector = vector

    @property
    def vector(self):
        """
        The vector to take the angle with.
        """
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def constraint(self, eqstate, index):
        """
        Returns the angle between the the node normal and the reference vector.
        """
        # TODO: indices polygon should become self.index
        indices_polygon = [model.structure.node_index[nbr] for nbr in self.polygon]
        polygon = eqstate.xyz[indices_polygon, :]
        normal = normal_polygon(polygon)
        return angle_vectors(normal, self.vector[index, :])
