from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal


class FaceGoal(Goal):
    """
    The base class for goals defined on the faces of a mesh.
    """

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's face key to an index in a structure.

        Parameters
        ----------
        structure :
            The mesh structure whose face ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's face(s).

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

        return self._index_from_key(structure.face_index)
