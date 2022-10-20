from jax_fdm.geometry import closest_point_on_segment
# from compas.geometry import closest_point_on_segment
from jax_fdm.goals.nodegoal import NodeLineGoal


class NodeSegmentGoal(NodeLineGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    @staticmethod
    def goal(target, prediction):
        """
        """
        return closest_point_on_segment(prediction, target)
