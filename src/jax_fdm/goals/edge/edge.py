import numpy as np
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["EdgeGoal"]


class EdgeGoal(Goal):
    """
    The base class for goals defined on the edges of a network.
    """

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "edges 2"]:
        """
        The structure's edge keys, the vocabulary an edge goal resolves against.

        Parameters
        ----------
        structure :
            The structure whose edge ordering defines the index.

        Returns
        -------
        keys_canonical :
            The structure's edge key pairs, one row per edge.
        """
        return structure.edges
