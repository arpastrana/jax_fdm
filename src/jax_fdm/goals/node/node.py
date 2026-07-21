from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["NodeGoal"]


class NodeGoal(Goal):
    """
    The base class for goals defined on the nodes of a network.
    """

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's node key to an index in a structure.

        Parameters
        ----------
        structure :
            The structure whose node ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's node(s).

        Raises
        ------
        TypeError
            If the structure is a mesh structure. Nodes are network vocabulary;
            on a mesh, use the goal's Vertex* counterpart.
        """
        if isinstance(structure, EquilibriumMeshStructure):
            raise TypeError(
                f"{type(self).__name__} targets network nodes. "
                "Use its Vertex* counterpart on a mesh.",
            )

        return self._indices_from_keys(structure.nodes)
