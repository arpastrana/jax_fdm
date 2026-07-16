from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal


class NetworkGoal(Goal):
    """
    The base class for goals defined on a network as a whole.

    Parameters
    ----------
    key :
        The sentinel element key; a network goal aggregates the whole structure.
    target :
        The value the goal drives its aggregate quantity toward.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    A network goal spans the entire structure rather than one element, so it always
    carries the sentinel key ``-1`` and is never grouped into a collection.
    """

    def __init__(
        self,
        key: int = -1,
        target: float | Float[Array, "..."] = 0.0,
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)
        # A network goal aggregates the whole structure, so it always carries the
        # sentinel key -1 and is never grouped with peers into a collection.
        self.is_collectible = False

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int:
        """
        Return the sentinel index shared by all network goals.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure the goal is bound to.

        Returns
        -------
        index :
            The sentinel index ``-1``, since the goal spans the whole network.
        """
        return -1
