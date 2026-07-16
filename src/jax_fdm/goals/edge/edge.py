from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class EdgeGoal(Goal):
    """
    The base class for goals defined on the edges of a network.
    """

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's edge key to an index in a structure.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose edge ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's edge(s).
        """
        return self._index_from_key(structure.edge_index)
