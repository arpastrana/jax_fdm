import numpy as np

from jax_fdm.goals.vertex import VertexNormalAngleGoal


class VertexTangentAngleGoal(VertexNormalAngleGoal):
    """
    Reach a target value for the angle formed by the vertex tangent and a reference vector.

    The vertex tangent is calculated as 90º minus the vertex normal.
    """
    def __init__(self, key, vector, target, weight=1.0):
        super().__init__(key=key, vector=vector, target=target, weight=weight)

    def prediction(self, eqstate, index):
        """
        Returns the angle between the vertex tangent and the reference vector.
        """
        angle_normal = super().prediction(eqstate, index)

        angle_tangent = np.pi * 0.5 - angle_normal

        return angle_tangent
