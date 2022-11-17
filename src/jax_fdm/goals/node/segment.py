from jax_fdm.geometry import closest_point_on_segment

from jax_fdm.goals.node import NodeLineGoal


class NodeSegmentGoal(NodeLineGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    @staticmethod
    def goal(target, prediction):
        """
        The closes point on the target seegment.
        """
        return closest_point_on_segment(prediction, target)
