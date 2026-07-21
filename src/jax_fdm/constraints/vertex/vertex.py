from jax_fdm.constraints.constraint import Constraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure

__all__ = ["VertexConstraint"]


class VertexConstraint(Constraint):
    """
    The base class for constraints defined on the vertices of a mesh.
    """

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the constraint's vertex key to an index in a structure.

        Parameters
        ----------
        structure :
            The mesh structure whose vertex ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the constraint's vertex(es).

        Raises
        ------
        TypeError
            If the structure is not a mesh structure. Vertices are mesh
            vocabulary; on a network, use the constraint's Node* counterpart.
        """
        if not isinstance(structure, EquilibriumMeshStructure):
            raise TypeError(
                f"{type(self).__name__} targets mesh vertices. "
                "Use its Node* counterpart on a network.",
            )

        return self._indices_from_keys(structure.vertices)
