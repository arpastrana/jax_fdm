from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.goals import Goal


class FaceGoal(Goal):
    """
    Base class for all constraints that pertain to a face in a mesh.
    """
    def index_from_model(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> int | tuple[int, ...]:
        """
        The index of the face in a structure.
        """
        return self._index_from_key(structure.face_index)
