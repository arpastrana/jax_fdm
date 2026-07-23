import numpy as np
from jaxtyping import Int

from jax_fdm.constraints.constraint import Constraint
from jax_fdm.equilibrium import EquilibriumStructure

__all__ = ["EdgeConstraint"]


class EdgeConstraint(Constraint):
    """
    The base class for constraints defined on the edges of a network.
    """

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "edges 2"]:
        """
        The structure's edge keys, the vocabulary an edge constraint resolves against.

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
