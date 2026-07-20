from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import Goal
from jax_fdm.goals.goal import TargetLike


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
    A mesh goal spans the entire structure rather than one element, so it is an
    aggregate that always carries the sentinel key ``-1``.
    """

    is_aggregate = True

    def __init__(
        self,
        key: int = -1,
        target: TargetLike = 0.0,
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int:
        """
        Return the sentinel index shared by all mesh goals.

        Parameters
        ----------
        structure :
            The mesh structure the goal is bound to.

        Returns
        -------
        index :
            The sentinel index ``-1``, since the goal spans the whole mesh.
        """
        return -1
