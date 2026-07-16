from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class NodeGoal(Goal):
    """
    Base class for all goals that pertain to the node of a network.
    """

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        The index of the node in a structure.
        """
        return self._index_from_key(structure.node_index)
