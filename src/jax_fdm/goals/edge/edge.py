from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class EdgeGoal(Goal):
    """
    Base class for all goals that pertain to an edge of a network.
    """
    def index_from_model(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int | tuple[int, ...]:
        """
        The index of the edge key in an equilibrium structure.
        """
        return self._index_from_key(structure.edge_index)
