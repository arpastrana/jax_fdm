from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class NetworkConstraint(Constraint):
    """
    Base class for all constraints that pertain to all the edges or all the nodes
    of a network.
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
        The index of the constraint key in a structure.
        """
        return -1
