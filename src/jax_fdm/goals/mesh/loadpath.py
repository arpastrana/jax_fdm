from jax_fdm.goals import NetworkLoadPathGoal


class MeshLoadPathGoal(NetworkLoadPathGoal):
    """
    Make the total load path of a mesh to reach a target magnitude.
    """
    def __init__(self, target=0.0, weight=1.0):
        super().__init__(key=-1, target=target, weight=weight)
