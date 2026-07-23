from collections.abc import Sequence
from typing import ClassVar
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import datastructure_state
from jax_fdm.equilibrium import indices_from_keys
from jax_fdm.goals.state import GoalState

__all__ = ["Goal", "as_key", "as_target"]

# What the target converters accept: a scalar, an existing array, or any nesting
# of float sequences (the converters run the input through jnp.asarray). COMPAS
# geometry objects are deliberately excluded; convert them to plain lists at the
# call site.
TargetLike: TypeAlias = (
    float | Float[Array, "..."] | Sequence[float] | Sequence[Sequence[float]]
)

# A goal's key: one element key (a node/vertex/face int, or an edge (u, v) pair),
# or a sequence of them for an aggregate goal. Aggregate keys are any sequence,
# a list or a tuple alike, since a single edge key is itself a two-int tuple and
# the aggregate-vs-single distinction is the goal's is_aggregate flag, never the
# key's Python type.
KeyLike: TypeAlias = int | tuple[int, int] | Sequence[int] | Sequence[tuple[int, int]]


def as_target(target: TargetLike) -> Float[Array, "..."]:
    """
    Coerce a goal's target to a JAX array.

    Parameters
    ----------
    target :
        The target value: a scalar, an xyz vector, or an arbitrary per-element
        array.

    Returns
    -------
    target :
        The target as a JAX array, its shape unchanged.

    Notes
    -----
    A single converter for every goal, so the field type it feeds the pytree is
    uniform. A goal holds one element's value, unbatched: a collection stacks
    same-type goals along a new leading axis at `tree_stack`. A goal's target
    shape (a scalar, an xyz vector, or an arbitrary array) is whatever its
    `prediction` returns; the only invariant, that the goal and prediction shapes
    agree, is checked at evaluation, not coerced here.
    """
    return jnp.asarray(target, dtype=DTYPE_JAX)


def as_weight(weight: float | Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Coerce a goal's weight to a JAX array.

    Parameters
    ----------
    weight :
        The importance of the goal.

    Returns
    -------
    weight :
        The scalar weight as a JAX array.

    Notes
    -----
    A weight is one scalar per element, stored unbatched like the goal's other
    leaves. The error term composes it with the prediction, so it is the error
    term, not this converter, that lifts the weight to the prediction's rank and
    broadcasts it down the feature axis.
    """
    return jnp.asarray(weight, dtype=DTYPE_JAX)


def as_key(key: KeyLike) -> Int[Array, "..."]:
    """
    Coerce a goal's key to a JAX array.

    Parameters
    ----------
    key :
        The element key of a per-element goal (a node/vertex/face int, or an
        edge (u, v) pair), or the sequence of keys an aggregate goal reduces
        over (a list or a tuple alike, or the ``-1`` whole-structure sentinel).

    Returns
    -------
    key :
        The key as a JAX integer array, its shape unchanged from the input: one
        element's key for a per-element goal, the whole selection for an
        aggregate.

    Notes
    -----
    One converter for every goal, so the field it feeds the pytree is uniform
    and no goal declares its own: an author sets `is_aggregate` and inherits the
    base key field unchanged. The per-element-vs-aggregate distinction is that
    flag alone, never the key's Python type, so this converter neither knows nor
    needs to know which it is coercing.

    The key is a dynamic array leaf, not a static field. Two goals of one type
    differing only in their key then share a pytree structure, so `tree_stack`
    groups them by stacking leaves along a new leading axis. The key stays a
    compile-time constant (a goal is closed over, never an argument to the jitted
    step), so resolving it against the structure's canonical ordering is a
    trace-time constant fold.

    An off-contract key is left to fail on its own terms: a one-shot iterator
    is not a valid JAX array type and `jnp.asarray` rejects it here, before any
    tree flatten; a list handed to a per-element goal stores cleanly but trips a
    `vmap` size mismatch at evaluation, once the element count is known.
    """
    return jnp.asarray(key, dtype=DTYPE_INT_JAX)


# ==========================================================================
# Base goal
# ==========================================================================


class Goal(eqx.Module):
    """
    The base class for all goals, targets an equilibrium quantity reaches.

    Attributes
    ----------
    key :
        The key of the element the goal acts on; a sequence of keys (a list or
        a tuple) only for aggregate goals.
    target :
        The value the goal drives its quantity of interest toward.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    A goal is a registered JAX pytree (an equinox Module): its `key`, `target`,
    and `weight` are dynamic array leaves. Construction stores one element's
    values unbatched, so a goal is a single-element object; a collection stacks
    same-type goals along a new leading axis and the error term `vmap`s the goal
    over it. The key is resolved to an index against an equilibrium structure at
    evaluation time. Subclasses supply the quantity of interest via `prediction`;
    the target's shape is whatever that prediction returns (a scalar, an xyz
    vector, or an arbitrary per-element array), and `__call__` checks the two
    agree.

    A per-element goal takes exactly one element key; to act on many elements,
    create one goal per element and let collections vectorize them. Only
    aggregate goals, which declare `is_aggregate = True`, accept a list of
    keys.
    """

    key: Int[Array, "..."] = eqx.field(converter=as_key)  # pyright: ignore[reportAssignmentType]
    target: Float[Array, "..."] = eqx.field(converter=as_target)
    weight: Float[Array, "..."] = eqx.field(converter=as_weight, default=1.0)  # pyright: ignore[reportAssignmentType]

    # An aggregate goal reduces over many elements (a key list, or the whole
    # structure) in one prediction call: its key is the whole list, resolved to a
    # whole index row that one prediction reads, and the goal is never grouped
    # with same-type peers into a vectorized collection since it is already a
    # batch of its own. A ClassVar so it stays a plain class attribute a subclass
    # overrides, out of the pytree.
    is_aggregate: ClassVar[bool] = False

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
        index: Int[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        Extract the quantity of interest for one element from an equilibrium state.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the quantity from.
        structure :
            The structure the goal is evaluated against, so a goal can read whole
            trace-time constants (connectivity, topology) from it. Most goals
            ignore it.
        index :
            The element index, one for a per-element goal or the whole index row
            for an aggregate goal. A goal carrying an extra per-element array
            reads it off `self`.

        Returns
        -------
        prediction :
            The current value of the quantity of interest.
        """
        raise NotImplementedError

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "elements ..."]:
        """
        The structure's canonical keys for this goal's vocabulary.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        keys_canonical :
            The node, edge, vertex, or face key ordering the goal's key is
            resolved against, one entry per element: a node/vertex/face key, or
            an edge key pair.

        Notes
        -----
        The only per-subclass part of key resolution: a subclass names the
        canonical ordering its key belongs to (nodes, vertices, faces, or edge
        pairs) and raises if the structure is the wrong kind. The resolution
        itself lives once in `index`, so an author who defines a goal on a new
        vocabulary overrides this hook alone.
        """
        raise NotImplementedError

    def index(self, structure: EquilibriumStructure) -> Int[Array, "..."]:
        """
        The goal's element index, resolved against the structure.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            A single index for a per-element goal, or the whole index row for an
            aggregate goal, which reads its whole selection in one prediction.

        Notes
        -----
        The real work of key resolution: a subclass names its canonical keys via
        `keys_from_structure`, and this resolves the goal's key against them in
        one `indices_from_keys` call, never a per-key loop. The key stays a
        compile-time constant (a goal is closed over, never a jitted argument),
        so the resolution folds to a constant at trace time. The key's own shape
        distinguishes a per-element goal from an aggregate, so there is no
        aggregate special case here.
        """
        keys_canonical = self.keys_from_structure(structure)

        return indices_from_keys(keys_canonical, self.key)

    def __call__(
        self,
        eqstate: EquilibriumState,
        structure: EquilibriumStructure,
    ) -> GoalState:
        """
        Evaluate the goal for one element against an equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to evaluate the goal on.
        structure :
            The structure whose element ordering resolves the goal's index.

        Returns
        -------
        goal_state :
            The goal state bundling the reference value, the prediction, and the
            weight for this one element, in the raw shape its hooks return.

        Raises
        ------
        ValueError
            If the goal and prediction shapes disagree, typically because a
            scalar goal's prediction returned more than one value per element.

        Notes
        -----
        A goal holds one element, so this is the linear per-element body: resolve
        the index, read the prediction, form the reference, check the two shapes
        agree, and bundle the state. A standalone call returns one element's
        unbatched state.

        To evaluate a group of same-type goals, stack them into one pytree and
        `vmap`, exactly as one maps any equinox module over a batch::

            stacked = tree_stack(goals)
            states = jax.vmap(lambda g: g(eqstate, structure))(stacked)

        `tree_stack` is a convenience, not a requirement: a plain
        ``tree_map(lambda *leaves: jnp.stack(leaves), *goals)`` builds the same
        stacked pytree, so the idiom holds for any stack of a goal's leaves.
        """
        index = self.index(structure)
        prediction = self.prediction(eqstate, structure, index)
        goal = self.goal(self.target, prediction)

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
            The goal state bundling the reference value, the prediction, and the
            weight for this one element.

        Notes
        -----
        A convenience for prototyping a goal on the high-level COMPAS layer: it
        reads the equilibrium state and structure off the datastructure and
        evaluates the goal against them in one call.
        """
        equilibrium = datastructure_state(datastructure, sparse)

        return self(equilibrium.eq_state, equilibrium.structure)
