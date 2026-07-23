import numpy as np
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal

__all__ = ["FaceGoal"]


class FaceGoal(Goal):
    """
    The base class for goals defined on the faces of a mesh.
    """

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "faces"]:
        """
        The structure's face keys, the vocabulary a face goal resolves against.

        Parameters
        ----------
        structure :
            The mesh structure whose face ordering defines the index.

        Returns
        -------
        keys_canonical :
            The mesh structure's face keys, one per face.

        Raises
        ------
        TypeError
            If the structure is not a mesh structure. Faces are mesh vocabulary;
            a network has no faces to aim a goal at.
        """
        if not isinstance(structure, EquilibriumMeshStructure):
            raise TypeError(
                f"{type(self).__name__} targets mesh faces. Networks have no faces.",
            )

        return structure.face_keys
