from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.goals import Goal


class MeshGoal(Goal):
    """
    The base class for goals defined on a mesh as a whole.

    Parameters
    ----------
    key :
        The sentinel element key; a mesh goal aggregates the whole structure.
    target :
        The value the goal drives its aggregate quantity toward.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    A mesh goal spans the entire structure rather than one element, so it always
    carries the sentinel key ``-1`` and is never grouped into a collection.
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
        Return the sentinel index shared by all mesh goals.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure the goal is bound to.

        Returns
        -------
        index :
            The sentinel index ``-1``, since the goal spans the whole mesh.
        """
        return -1
