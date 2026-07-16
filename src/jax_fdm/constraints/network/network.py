from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class NetworkConstraint(Constraint):
    """
    The base class for constraints defined on a network as a whole.

    Parameters
    ----------
    bound_low :
        The lower bound applied to every element. If None, unbounded below.
    bound_up :
        The upper bound applied to every element. If None, unbounded above.

    Notes
    -----
    A network constraint spans all edges or nodes at once rather than one element,
    so it always carries the sentinel key ``-1``.
    """

    def __init__(
        self,
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ) -> None:
        super().__init__(key=-1, bound_low=bound_low, bound_up=bound_up)

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int:
        """
        Return the sentinel index shared by all network constraints.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure the constraint is bound to.

        Returns
        -------
        index :
            The sentinel index ``-1``, since the constraint spans the whole network.
        """
        return -1
