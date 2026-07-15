from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class EdgeGoal(Goal):
    """
    Base class for all goals that pertain to an edge of a network.
    """
    def __init__(
        self,
        key: int | tuple[int, int] | list,
        target: float | Float[Array, "..."],
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)

    def index_from_model(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int | tuple[int, ...]:
        """
        The index of the edge key in an equilibrium structure.
        """
        return self._index_from_key(structure.edge_index)
