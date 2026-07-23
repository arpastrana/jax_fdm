import equinox as eqx
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import Goal
from jax_fdm.goals.goal import as_key
from jax_fdm.goals.goal import as_target

__all__ = ["NetworkGoal"]


class NetworkGoal(Goal):
    """
    The base class for goals defined on a network as a whole.

    Parameters
    ----------
    target :
        The value the goal drives its aggregate quantity toward.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    A network goal spans the entire structure rather than one element, so it is an
    aggregate that always carries the sentinel key ``-1``. The key is fixed with
    ``init=False``, so it is hidden from the constructor rather than exposed as a
    passable-but-ignored argument, and the whole network-goal family constructs
    from a target and a weight alone.
    """

    is_aggregate = True

    key: Int[Array, "..."] = eqx.field(converter=as_key, default=-1, init=False)  # pyright: ignore[reportAssignmentType]
    target: Float[Array, "..."] = eqx.field(converter=as_target, default=0.0)  # pyright: ignore[reportAssignmentType]

    def index(
        self,
        structure: EquilibriumStructure,
    ) -> Int[Array, "..."]:
        """
        Return the sentinel index shared by all network goals.

        Parameters
        ----------
        structure :
            The structure the goal is bound to.

        Returns
        -------
        index :
            The sentinel index ``-1``, since the goal spans the whole network and
            there is no element key to resolve.

        Notes
        -----
        A whole-structure aggregate has nothing to resolve, so it overrides
        `index` to short-circuit the key resolution the per-element goals run,
        returning the sentinel unchanged. Every network-goal prediction reads the
        whole structure directly and ignores this index.
        """
        return self.key
