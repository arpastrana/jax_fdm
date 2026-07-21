from collections.abc import Sequence
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import equilibrium_state_from_datastructure
from jax_fdm.equilibrium import indices_from_keys
from jax_fdm.goals.state import GoalState

__all__ = ["Goal", "ScalarGoal", "VectorGoal"]

# What the target setters accept: a scalar, an existing array, or any nesting
# of float sequences (the setters run the input through jnp.asarray + reshape).
# COMPAS geometry objects are deliberately excluded; convert them to plain
# lists at the call site.
TargetLike: TypeAlias = (
    float | Float[Array, "..."] | Sequence[float] | Sequence[Sequence[float]]
)

# ==========================================================================
# Base goal
# ==========================================================================


class Goal:
    """
    The base class for all goals, targets an equilibrium quantity reaches.

    Parameters
    ----------
    key :
        The key of the element the goal acts on; a list of keys only for
        aggregate goals.
    target :
        The value the goal drives its quantity of interest toward.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    A goal is stateless: construction stores the key, target, and weight, and
    `__call__` resolves the key to an index against an equilibrium structure at
    evaluation time. Subclasses supply the quantity of interest via `prediction`
    and mix in [ScalarGoal][jax_fdm.goals.goal.ScalarGoal] or
    [VectorGoal][jax_fdm.goals.goal.VectorGoal] for the target's shape.

    A per-element goal takes exactly one element key; to act on many elements,
    create one goal per element and let collections vectorize them. Only
    aggregate goals, which declare `is_aggregate = True`, accept a list of
    keys.
    """

    # An aggregate goal reduces over many elements (a key list, or the whole
    # structure) in one prediction call: __call__ feeds vmap the index row
    # unsplit, and the goal is never grouped with same-type peers into a
    # vectorized collection since it is already a batch of its own.
    is_aggregate: bool = False

    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        target: TargetLike,
        weight: float | Float[Array, "..."] = 1.0,
    ) -> None:
        self._key: int | tuple[int, int] | list[int] | list[tuple[int, int]] | None = (
            None
        )
        self._weight: Float[Array, "elements 1"]
        self._target: Float[Array, "..."]

        self.key = key
        self.weight = weight
        self.target = target

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
        if isinstance(key, list):
            if not key:
                raise ValueError(
                    f"{type(self).__name__} got an empty key list. "
                    "Pass at least one element key.",
                )
            if not self.is_aggregate and not getattr(self, "_iscollection", False):
                raise TypeError(
                    f"{type(self).__name__} takes a single element key, got a "
                    f"list of {len(key)}. Create one goal per element "
                    "(collections vectorize same-type goals automatically), or "
                    "use an aggregate goal for group quantities.",
                )
        self._key = key

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
    def target(self, target: TargetLike) -> None:
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
        structure: EquilibriumStructure,
        payload: Int[Array, ""] | tuple[Array, ...],
    ) -> Float[Array, "..."]:
        """
        Extract the quantity of interest for one element from an equilibrium state.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the quantity from.
        structure :
            The structure the goal is evaluated against, held out of the vmap so
            a goal can read whole trace-time constants (connectivity, topology)
            from it. Most goals ignore it.
        payload :
            The per-element slice vmap maps at axis 0. The index of the element
            for a plain goal, or a tuple the goal unpacks when it carries an extra
            per-element array (see `operand`).

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

    def _indices_from_keys(
        self,
        keys_canonical: Int[np.ndarray, "elements ..."],
    ) -> int | tuple[int, ...]:
        """
        Resolve the goal's key(s) to positions in a canonical key ordering.

        Parameters
        ----------
        keys_canonical :
            The structure's canonical key ordering (nodes, vertices, faces, or
            edge pairs), one row per element.

        Returns
        -------
        index :
            A single index for a scalar key, or a tuple of indices for a list key.
        """
        key = self.key
        if key is None:
            raise ValueError(f"{type(self).__name__} has no key to resolve.")

        resolved = indices_from_keys(keys_canonical, key)
        if isinstance(key, list):
            return tuple(int(index) for index in resolved)

        return int(resolved[0])

    def indices(self, structure: EquilibriumStructure) -> Int[np.ndarray, "elements"]:
        """
        The goal's element index as a one-dimensional array, ready for vmap.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            One entry per element for a per-element goal; a single batch-of-one
            row for an aggregate goal, so vmap maps its whole index at once.
        """
        index = np.atleast_1d(np.asarray(self.index_from_structure(structure)))
        if self.is_aggregate:
            index = index[np.newaxis, :]

        return index

    def operand(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "elements"] | tuple[Array, ...]:
        """
        The per-element payload vmap maps at axis 0.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        payload :
            The index array by default. A goal carrying an extra per-element
            array overrides this to return a tuple, e.g. an angle goal returns
            ``(self.indices(structure), self.vector)``; `prediction` unpacks it.
            Every leaf is collection-ordered, so vmap zips row k of each into
            call k.
        """
        return self.indices(structure)

    def __call__(
        self,
        eqstate: EquilibriumState,
        structure: EquilibriumStructure,
    ) -> GoalState:
        """
        Evaluate the goal against an equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to evaluate the goal on.
        structure :
            The structure whose element ordering resolves the goal's index.

        Returns
        -------
        goal_state :
            The goal state bundling the reference values, the predictions, and the
            weights, vmapped over the goal's elements.

        Raises
        ------
        ValueError
            If the number of target rows does not match the number of elements,
            or if the goal and prediction shapes disagree, typically because a
            scalar goal's prediction returned more than one value per element.
        """
        # One per-element payload is mapped at axis 0; eqstate and structure are
        # held out (in_axes=None). structure rides whole rather than batched, so
        # a goal may read trace-time constants off it and sparse matrices never
        # become vmap operands.
        payload = self.operand(structure)
        # payload is a numpy index array (or a tuple of mapped arrays); vmap maps
        # its leading axis to the per-element type the hook annotates.
        prediction = vmap(self.prediction, in_axes=(None, None, 0))(
            eqstate,
            structure,
            payload,  # pyright: ignore[reportArgumentType]
        )

        if self.target.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"{type(self).__name__}: {prediction.shape[0]} element(s) but "
                f"{self.target.shape[0]} target row(s). Pass one target per "
                "element.",
            )

        goal = vmap(self.goal)(self.target, prediction)

        # Flatten per-element shapes to one feature row only after the goal
        # hook, so a custom goal() sees the prediction's true per-element
        # shape; a scalar prediction may return () per element, which vmap
        # stacks to (elements,) and lands here as one value per row.
        prediction = jnp.reshape(prediction, (prediction.shape[0], -1))
        goal = jnp.reshape(goal, (goal.shape[0], -1))

        if goal.shape != prediction.shape:
            raise ValueError(
                f"{type(self).__name__}: goal shape {goal.shape} != prediction "
                f"shape {prediction.shape}. The prediction must return one value "
                "per element for a scalar goal, or one vector per element for a "
                "vector goal.",
            )

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)

    def evaluate(
        self,
        datastructure: FDNetwork | FDMesh,
        sparse: bool = True,
    ) -> GoalState:
        """
        Evaluate the goal directly on a datastructure, without an optimization.

        Parameters
        ----------
        datastructure :
            The network or mesh to read the equilibrium state from. Its geometry
            is used as-is; no form-finding is run.
        sparse :
            If True, assemble the equilibrium state with the sparse model.

        Returns
        -------
        goal_state :
            The goal state bundling the reference values, the predictions, and the
            weights.

        Notes
        -----
        A convenience for prototyping a goal on the high-level COMPAS layer: it
        builds the equilibrium state, structure, and model from the datastructure
        and evaluates the goal against them in one call.
        """
        equilibrium = equilibrium_state_from_datastructure(datastructure, sparse)

        return self(equilibrium.eq_state, equilibrium.structure)


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
    def target(self, target: TargetLike) -> None:
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
    def target(self, target: TargetLike) -> None:
        if isinstance(target, (int, float)):
            raise TypeError(
                f"{type(self).__name__} is a vector goal, so its target must be "
                "an xyz vector (or one per element), not a single number. "
                "Did you mean the scalar variant of this goal?",
            )
        self._target = jnp.reshape(jnp.asarray(target, dtype=DTYPE_JAX), (-1, 3))
