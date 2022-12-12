import numpy as np

from jax_fdm.constraints.node import NodeNormalAngleConstraint


class NodeTangentAngleConstraint(NodeNormalAngleConstraint):
    """
    Constraints the angle formed by the node tangent and a vector between a lower and an upper bound.

    The node tangent is calculated as 90º minus the node normal.
    """
    def __init__(self, key, vector, bound_low=None, bound_up=None):
        super().__init__(key=key, vector=vector, bound_low=bound_low, bound_up=bound_up)

    def constraint(self, eqstate, index):
        """
        Returns the curvature at a node based on the xyz coordinates of its one-hop neighborhood.
        """
        normal = super().constraint(eqstate, index)

        return np.pi * 0.5 - normal
