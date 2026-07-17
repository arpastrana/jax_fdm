from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import GoalState

# ==========================================================================
# Base goal
# ==========================================================================


class Goal:
    """
    The base class for all goals, targets an equilibrium quantity reaches.

    Parameters
    ----------
    key :
        The key or keys of the element(s) the goal acts on.
    target :
        The value the goal drives its quantity of interest toward.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    A goal is initialized in two phases: construction stores the key, target, and
    weight, then `init` resolves the key to an index against an equilibrium
    structure before any prediction runs. Subclasses supply the quantity of
    interest via `prediction` and mix in [ScalarGoal][jax_fdm.goals.goal.ScalarGoal]
    or [VectorGoal][jax_fdm.goals.goal.VectorGoal] for the target's shape.
    """

    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        target: float | Float[Array, "..."],
        weight: float | Float[Array, "..."] = 1.0,
    ) -> None:
        self._key: int | tuple[int, int] | list[int] | list[tuple[int, int]] | None = (
            None
        )
        self._weight: Float[Array, "elements 1"]
        self._target: Float[Array, "..."]
        # set in init() from the equilibrium structure, before any prediction runs
        self._index: Int[np.ndarray, "elements"]

        self.key = key
        self.weight = weight
        self.target = target

        self.is_collectible = True

    @property
    def key(self) -> int | tuple[int, int] | list[int] | list[tuple[int, int]] | None:
        """
        The key of an element in a network.
        """
        return self._key

    @key.setter
    def key(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
    ) -> None:
        # A single-goal Collection re-wraps an already-list key as [[...]] when
        # it reconstructs the goal; unwrap that extra nesting so an aggregate
        # goal (e.g. NodesColinearGoal) keeps its flat list of element keys.
        if isinstance(key, list) and len(key) == 1 and isinstance(key[0], list):
            key = key[0]
        self._key = key

    @property
    def index(self) -> Int[np.ndarray, "elements"]:
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        return self._index

    @index.setter
    def index(self, index: int | tuple[int, ...] | Int[np.ndarray, "elements"]) -> None:
        if isinstance(index, int):
            index = (index,)
        self._index = np.asarray(index)

    @property
    def weight(self) -> Float[Array, "elements 1"]:
        """
        The importance of the goal.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float | Float[Array, "..."]) -> None:
        self._weight = jnp.reshape(jnp.asarray(weight, dtype=DTYPE_JAX), (-1, 1))

    @property
    def target(self) -> Float[Array, "..."]:
        """
        The target to achieve.
        """
        raise NotImplementedError

    @target.setter
    def target(self, target: float | Float[Array, "..."]) -> None:
        # Concrete goals provide the setter via the ScalarGoal / VectorGoal
        # mixins; the base only declares the contract so that Goal.__init__ can
        # assign self.target without tripping a read-only property.
        raise NotImplementedError

    def goal(
        self,
        target: Float[Array, "..."],
        prediction: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        The reference value the prediction is compared against.

        Parameters
        ----------
        target :
            The goal's target value.
        prediction :
            The current value of the quantity of interest.

        Returns
        -------
        goal :
            The reference value. The base goal returns the target unchanged;
            subclasses may combine it with the prediction (e.g. projections).
        """
        return target

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        Extract the quantity of interest for one element from an equilibrium state.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the quantity from.
        index :
            The index of the element within the equilibrium state.

        Returns
        -------
        prediction :
            The current value of the quantity of interest.
        """
        raise NotImplementedError

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's key to an index in an equilibrium structure.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the goal's element(s).
        """
        raise NotImplementedError

    def _index_from_key(self, key_index: dict[Any, int]) -> int | tuple[int, ...]:
        """
        Look up the goal's key in an element-to-index mapping.

        Parameters
        ----------
        key_index :
            The mapping from element key to index.

        Returns
        -------
        index :
            A single index for a scalar key, or a tuple of indices for a list key.
        """
        if isinstance(self.key, list):
            return tuple(key_index[k] for k in self.key)
        return key_index[self.key]

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Bind the goal to a structure by resolving its key to an index.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose element ordering defines the index.

        Notes
        -----
        Must be called once before the goal is evaluated; it populates the index
        that `prediction` reads.
        """
        self.index = self.index_from_structure(structure)

    def __call__(self, eqstate: EquilibriumState) -> GoalState:
        """
        Evaluate the goal against an equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to evaluate the goal on.

        Returns
        -------
        goal_state :
            The goal state bundling the reference values, the predictions, and the
            weights, vmapped over the goal's elements.

        Raises
        ------
        ValueError
            If the goal and prediction shapes disagree, typically because a scalar
            prediction dropped its trailing axis.
        """
        # self.index is a numpy index array populated in init and mapped to a
        # jax scalar by vmap
        prediction = vmap(self.prediction, in_axes=(None, 0))(eqstate, self.index)  # pyright: ignore[reportArgumentType]
        goal = vmap(self.goal)(self.target, prediction)

        if goal.shape != prediction.shape:
            raise ValueError(
                f"{type(self).__name__}: goal shape {goal.shape} != prediction "
                f"shape {prediction.shape}. Scalar predictions must have shape "
                "(1,); wrap the prediction's return value with jnp.atleast_1d.",
            )

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)


# ==========================================================================
# Base goal for a scalar quantity
# ==========================================================================


class ScalarGoal:
    """
    A mixin for goals whose target is a scalar quantity per element.

    Notes
    -----
    Reshapes the target to a column so each element carries one scalar value.
    """

    @property
    def target(self) -> Float[Array, "elements 1"]:
        """
        The scalar target value of each element.
        """
        return self._target

    @target.setter
    def target(self, target: float | Float[Array, "..."]) -> None:
        values = [target] if isinstance(target, (int, float)) else target
        self._target = jnp.reshape(jnp.asarray(values, dtype=DTYPE_JAX), (-1, 1))


# ==========================================================================
# Base goal for vector quantities
# ==========================================================================


class VectorGoal:
    """
    A mixin for goals whose target is a 3D vector quantity per element.

    Notes
    -----
    Reshapes the target so each element carries one xyz vector.
    """

    @property
    def target(self) -> Float[Array, "elements 3"]:
        """
        The 3D vector target of each element.
        """
        return self._target

    @target.setter
    def target(self, target: Float[Array, "..."]) -> None:
        self._target = jnp.reshape(jnp.asarray(target, dtype=DTYPE_JAX), (-1, 3))
