from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class NodeGoal(Goal):
    """
    The base class for goals defined on the nodes of a network.
    """

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's node key to an index in a structure.

        Parameters
        ----------
        model :
            The equilibrium model.
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

        return self._index_from_key(structure.node_index)
