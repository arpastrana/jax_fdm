import numpy as np
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["VertexGoal"]


class VertexGoal(Goal):
    """
    The base class for goals defined on the vertices of a mesh.
    """

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "vertices"]:
        """
        The structure's vertex keys, the vocabulary a vertex goal resolves against.

        Parameters
        ----------
        structure :
            The mesh structure whose vertex ordering defines the index.

        Returns
        -------
        keys_canonical :
            The mesh structure's vertex keys, one per vertex.

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

        return structure.vertices
