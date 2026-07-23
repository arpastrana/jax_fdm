import numpy as np
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["NodeGoal"]


class NodeGoal(Goal):
    """
    The base class for goals defined on the nodes of a network.
    """

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "nodes"]:
        """
        The structure's node keys, the vocabulary a node goal resolves against.

        Parameters
        ----------
        structure :
            The structure whose node ordering defines the index.

        Returns
        -------
        keys_canonical :
            The structure's node keys, one per node.

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

        return structure.nodes
