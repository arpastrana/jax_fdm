from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.goals import NetworkLoadPathGoal


class MeshLoadPathGoal(NetworkLoadPathGoal):
    """
    Drive the total load path of a mesh toward a target magnitude.

    Notes
    -----
    A thin wrapper of :class:`NetworkLoadPathGoal` that fixes the key to the mesh
    sentinel.
    """

    def __init__(
        self,
        target: float | Float[Array, "..."] = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=-1, target=target, weight=weight)
