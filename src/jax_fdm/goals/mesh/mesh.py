from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.goals import Goal


class MeshGoal(Goal):
    """
    Base class for all goals that pertain to a subset of the nodes, faces and
    edges of a mesh.
    """

    def __init__(
        self,
        key: int = -1,
        target: float | Float[Array, "..."] = 0.0,
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)
        # A mesh goal aggregates the whole structure, so it always carries the
        # sentinel key -1 and is never grouped with peers into a collection.
        self.is_collectible = False

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> int:
        """
        The index of the goal key in a structure.
        """
        return -1
