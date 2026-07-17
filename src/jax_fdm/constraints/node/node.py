from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class NodeConstraint(Constraint):
    """
    The base class for constraints defined on the nodes of a network.
    """

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the constraint's node key to an index in a structure.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose node ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the constraint's node(s).

        Raises
        ------
        TypeError
            If the structure is a mesh structure. Nodes are network vocabulary;
            on a mesh, use the constraint's Vertex* counterpart.
        """
        if isinstance(structure, EquilibriumMeshStructure):
            raise TypeError(
                f"{type(self).__name__} targets network nodes. "
                "Use its Vertex* counterpart on a mesh.",
            )

        return self._index_from_key(structure.node_index)
