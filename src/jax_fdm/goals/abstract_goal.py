from abc import abstractmethod
from abc import abstractproperty

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import GoalState

# ==========================================================================
# Abstract goal
# ==========================================================================

class AbstractGoal:

    @abstractmethod
    def __call__(self, eqstate: EquilibriumState) -> GoalState:
        """
        Return the current goal state.
        """
        raise NotImplementedError

    @abstractproperty
    def key(self) -> int | tuple[int, int] | list:
        """
        The key of an element in a network.
        """
        raise NotImplementedError

    @abstractproperty
    def index(self) -> Int[Array, "..."]:
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        raise NotImplementedError

    @abstractproperty
    def weight(self) -> float | Float[Array, "..."]:
        """
        The importance of the goal.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, eq_state: EquilibriumState) -> Float[Array, "..."]:
        """
        The current reference value in the equilibrium state.
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, prediction: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        The target to achieve.
        """
        raise NotImplementedError
