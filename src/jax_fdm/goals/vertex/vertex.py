from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["VertexGoal"]


class VertexGoal(Goal):
    """
    The base class for goals defined on the vertices of a mesh.
    """

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's vertex key to an index in a structure.

        Parameters
        ----------
        structure :
            The mesh structure whose vertex ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's vertex(es).

        Raises
        ------
        TypeError
            If the structure is not a mesh structure. Vertices are mesh
            vocabulary; on a network, use the goal's Node* counterpart.
        """
        if not isinstance(structure, EquilibriumMeshStructure):
            raise TypeError(
                f"{type(self).__name__} targets mesh vertices. "
                "Use its Node* counterpart on a network.",
            )

        return self._index_from_key(structure.vertex_index)
