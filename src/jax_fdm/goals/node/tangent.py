import numpy as np

from jax_fdm.goals.node import NodeNormalAngleGoal


class NodeTangentAngleGoal(NodeNormalAngleGoal):
    """
    Reach a target value for the angle formed by the node tangent and a reference vector.

    The node tangent is calculated as 90º minus the node normal.
    """
    def __init__(self, key, vector, target, weight=1.0):
        super().__init__(key=key, vector=vector, target=target, weight=weight)

    def prediction(self, eqstate, index):
        """
        Returns the angle between the node tangent and the reference vector.
        """
        normal = super().prediction(eqstate, index)

        return np.pi * 0.5 - normal
