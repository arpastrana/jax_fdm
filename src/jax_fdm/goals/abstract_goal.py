from abc import abstractmethod
from abc import abstractproperty


# ==========================================================================
# Abstract goal
# ==========================================================================

class AbstractGoal:

    @abstractmethod
    def __call__(self, eqstate):
        """
        Return the current goal state.
        """
        raise NotImplementedError

    @abstractproperty
    def key(self):
        """
        The key of an element in a network.
        """
        raise NotImplementedError

    @abstractproperty
    def index(self):
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        raise NotImplementedError

    @abstractproperty
    def weight(self):
        """
        The importance of the goal.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, eq_state):
        """
        The current reference value in the equilibrium state.
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, prediction):
        """
        The target to achieve.
        """
        raise NotImplementedError
