from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.goals import Goal


class FaceGoal(Goal):
    """
    The base class for goals defined on the faces of a mesh.
    """

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's face key to an index in a structure.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure whose face ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's face(s).
        """
        return self._index_from_key(structure.face_index)
