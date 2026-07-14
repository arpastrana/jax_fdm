from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class NodeGoal(Goal):
    """
    Base class for all goals that pertain to the node of a network.
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
        The index of the edge in a structure.
        """
        try:
            return structure.node_index[self.key]  # pyright: ignore[reportArgumentType]  # self.key may be a single node key or a list of node keys; the dict lookup dispatches at runtime via the TypeError below
        except TypeError:
            return tuple([structure.node_index[k] for k in self.key])  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]  # self.key is Optional and may be a single int/tuple by declaration but always set to a list here, since a scalar key would not raise TypeError above
