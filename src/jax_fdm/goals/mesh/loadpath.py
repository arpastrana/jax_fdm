from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.network.loadpath import NetworkLoadPathGoal

__all__ = ["MeshLoadPathGoal"]


class MeshLoadPathGoal(NetworkLoadPathGoal):
    """
    Drive the total load path of a mesh toward a target magnitude.

    Notes
    -----
    A thin wrapper of
    [NetworkLoadPathGoal][jax_fdm.goals.network.loadpath.NetworkLoadPathGoal]
    that fixes the key to the mesh sentinel.
    """

    def __init__(
        self,
        target: TargetLike = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(target=target, weight=weight)
