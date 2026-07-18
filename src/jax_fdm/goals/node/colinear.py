from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import colinearity_points
from jax_fdm.geometry import curvature_points
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.node import NodeGoal


class NodesColinearGoal(ScalarGoal, NodeGoal):
    """
    Minimize length-normalized colinearity energy for an ordered sequence of points.
    This goal favors solutions where points are evenly spaced.

    Notes
    -----
    - The goal applies to a *collection* of ordered nodes.
    - The start and end points are assumed to be fixed.
    """

    is_aggregate = True

    def __init__(self, key: list[int], weight: float = 1.0) -> None:
        super().__init__(key=key, target=0.0, weight=weight)

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "points"],
    ) -> Float[Array, ""]:
        """
        The length-normalized colinearity energy of the ordered nodes.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the node coordinates from.
        index :
            The indices of the ordered nodes.

        Returns
        -------
        prediction :
            The colinearity energy, zero when the nodes are evenly spaced on a line.
        """
        P = eq_state.xyz[index, :]

        return colinearity_points(P)


class NodesCurvatureGoal(ScalarGoal, NodeGoal):
    """
    Minimize curvature energy (i.e., the turning angle) for an ordered sequence
    of points.

    Notes
    -----
    - The goal applies to a *collection* of ordered nodes.
    - The start and end points are assumed to be fixed.
    """

    is_aggregate = True

    def __init__(self, key: list[int], weight: float = 1.0) -> None:
        super().__init__(key=key, target=0.0, weight=weight)

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "points"],
    ) -> Float[Array, ""]:
        """
        The curvature energy of the ordered nodes.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the node coordinates from.
        index :
            The indices of the ordered nodes.

        Returns
        -------
        prediction :
            The curvature energy, the total turning angle along the sequence.
        """
        P = eq_state.xyz[index, :]

        return curvature_points(P)
