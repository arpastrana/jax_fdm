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
        try:
            return structure.face_index[self.key]  # pyright: ignore[reportArgumentType]  # self.key may be a single face key or a list of face keys; the dict lookup dispatches at runtime via the TypeError below
        except TypeError:
            return tuple([structure.face_index[k] for k in self.key])  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]  # self.key is Optional and may be a single int/tuple by declaration but always set to a list here, since a scalar key would not raise TypeError above
