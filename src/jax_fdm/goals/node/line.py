import numpy as np

from jax_fdm.geometry import closest_point_on_line

from jax_fdm.goals.node import NodePointGoal


class NodeLineGoal(NodePointGoal):
    """
    Pulls the position of a node to a target line ray.
    """
    @property
    def target(self):
        """
        The target to achieve
        """
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.reshape(target, (-1, 2, 3))

    @staticmethod
    def goal(target, prediction):
        """
        The closest point on the target line.
        """
        return closest_point_on_line(prediction, target)
