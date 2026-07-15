from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.goals import Goal


class VertexGoal(Goal):
    """
    Base class for all constraints that pertain to a vertex in a mesh.
    """
    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
        ) -> int | tuple[int, ...]:
        """
        The index of the vertex in a structure.
        """
        return self._index_from_key(structure.vertex_index)
