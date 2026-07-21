from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["EdgeGoal"]


class EdgeGoal(Goal):
    """
    The base class for goals defined on the edges of a network.
    """

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's edge key to an index in a structure.

        Parameters
        ----------
        structure :
            The structure whose edge ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's edge(s).
        """
        return self._indices_from_keys(structure.edges)
