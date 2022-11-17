import numpy as np

from jax_fdm.geometry import closest_point_on_plane

from jax_fdm.goals.node import NodePointGoal


class NodePlaneGoal(NodePointGoal):
    """
    Pulls the xyz position of a node to a target plane.
    """
    @property
    def target(self):
        """
        The target to achieve.
        """
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.reshape(target, (-1, 2, 3))

    @staticmethod
    def goal(target, prediction):
        """
        Calculate the closest point on the target plane given the current node coordinates.
        """
        return closest_point_on_plane(prediction, target)
