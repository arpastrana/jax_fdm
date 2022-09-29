import numpy as np

import jax.numpy as jnp

from jax_fdm.constraints.node import NodeConstraint

from jax_fdm.geometry import curvature_point_polygon


class NodeCurvatureConstraint(NodeConstraint):
    """
    Constraints the (discrete) curvature of a node based on its surrounding polygon of neighboring nodes.
    """
    def __init__(self, key, polygon, bound_low, bound_up):
        super().__init__(key, bound_low, bound_up)
        self.polygon = polygon
        self.index_polygon = None

    def init(self, model):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model)
        self.index_polygon = self.polygon_indices(model)

    def polygon_indices(self, model):
        """
        Obtains the indices of the polygon from a model.
        """
        index_max = max(self.index) + 1
        polygon = np.atleast_2d(self.polygon)
        index_polygon = np.zeros((index_max, polygon.shape[1]))
        for p, idx in zip(polygon, self.index):
            index_polygon[idx, :] = tuple([model.structure.node_index[nbr] for nbr in p])

        return jnp.array(index_polygon, dtype=jnp.int64)

    def constraint(self, eqstate, index):
        """
        Returns the curvature at a node based on the xyz coordinates of its one-hop neighborhood.
        """
        point = eqstate.xyz[index, :]
        index_polygon = self.index_polygon[index, :]
        polygon = eqstate.xyz[index_polygon, :]

        return curvature_point_polygon(point, polygon)
