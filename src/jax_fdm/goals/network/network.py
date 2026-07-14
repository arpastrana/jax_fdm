from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class NetworkGoal(Goal):
    """
    Base class for all goals that pertain to a subset of the nodes and edges of a network.
    """
    def __init__(
        self,
        key: int | tuple[int, int] | list = -1,
        target: float | Float[Array, "..."] = 0.0,
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)

    def index_from_model(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int:
        """
        The index of the goal key in a structure.
        """
        return -1
