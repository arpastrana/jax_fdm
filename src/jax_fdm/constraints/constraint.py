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

__all__ = ["Constraint", "as_key", "as_bound_low", "as_bound_up"]

# A constraint's key: one element key (a node/vertex/face int, or an edge (u, v)
# pair), or a sequence of them for an aggregate constraint. Aggregate keys are
# any sequence, a list or a tuple alike, since a single edge key is itself a
# two-int tuple and the aggregate-vs-single distinction is the constraint's
# is_aggregate flag, never the key's Python type.
KeyLike: TypeAlias = int | tuple[int, int] | Sequence[int] | Sequence[tuple[int, int]]

# What the bound converters accept: a scalar, an existing array, or None. None
# is the open side, mapped to the appropriate signed infinity.
BoundLike: TypeAlias = float | Float[Array, "..."] | None


def as_key(key: KeyLike) -> Int[Array, "..."]:
    """
    Coerce a constraint's key to a JAX array.

    Parameters
    ----------
    key :
        The element key of a per-element constraint (a node/vertex/face int, or
        an edge (u, v) pair), or the sequence of keys an aggregate constraint
        spans (a list or a tuple alike, or the ``-1`` whole-structure sentinel).

    Returns
    -------
    key :
        The key as a JAX integer array, its shape unchanged from the input: one
        element's key for a per-element constraint, the whole selection for an
        aggregate.

    Notes
    -----
    One converter for every constraint, so the field it feeds the pytree is
    uniform and no constraint declares its own: an author sets `is_aggregate` and
    inherits the base key field unchanged. The per-element-vs-aggregate
    distinction is that flag alone, never the key's Python type, so this converter
    neither knows nor needs to know which it is coercing.

    The key is a dynamic array leaf, not a static field. Two constraints of one
    type differing only in their key then share a pytree structure, so
    `tree_stack` groups them by stacking leaves along a new leading axis. The key
    stays a compile-time constant (a constraint is closed over, never an argument
    to the jitted step), so resolving it against the structure's canonical
    ordering is a trace-time constant fold.

    A per-element constraint handed a list of keys stores it cleanly, then trips
    a `vmap` size mismatch at evaluation once the element count is known: like a
    goal, a constraint declares its arity by what its `constraint` hook returns,
    checked at evaluation rather than guarded at construction.
    """
    return jnp.asarray(key, dtype=DTYPE_INT_JAX)


def as_bound_low(bound: BoundLike) -> Float[Array, "..."]:
    """
    Coerce a constraint's lower bound to a JAX array.

    Parameters
    ----------
    bound :
        The lower bound on the constrained quantity. None leaves it unbounded
        below.

    Returns
    -------
    bound :
        The bound as a JAX array. An unbounded side is negative infinity, so a
        missing bound never masquerades as a real one.

    Notes
    -----
    The lower and upper bounds take their own converters rather than one shared
    converter, since the open side maps to a different infinity for each: below
    is negative, above is positive. A bound is stored unbatched, one scalar per
    element, so `tree_stack` stacks a collection's bounds along the leading axis
    the optimizer reads back as ``lb`` and ``ub``.
    """
    if bound is None:
        bound = -jnp.inf

    return jnp.asarray(bound, dtype=DTYPE_JAX)


def as_bound_up(bound: BoundLike) -> Float[Array, "..."]:
    """
    Coerce a constraint's upper bound to a JAX array.

    Parameters
    ----------
    bound :
        The upper bound on the constrained quantity. None leaves it unbounded
        above.

    Returns
    -------
    bound :
        The bound as a JAX array. An unbounded side is positive infinity, so a
        missing bound never masquerades as a real one.

    Notes
    -----
    The twin of `as_bound_low` for the upper side; see there for why the two
    sides need separate converters.
    """
    if bound is None:
        bound = jnp.inf

    return jnp.asarray(bound, dtype=DTYPE_JAX)


# ==========================================================================
# Base constraint
# ==========================================================================


class Constraint(eqx.Module):
    """
    The base class for all constraints, bounds an equilibrium quantity must obey.

    Attributes
    ----------
    key :
        The key of the element the constraint acts on; a sequence of keys (a list
        or a tuple) only for aggregate constraints.
    bound_low :
        The lower bound on the constrained quantity. If None, unbounded below.
    bound_up :
        The upper bound on the constrained quantity. If None, unbounded above.

    Notes
    -----
    A constraint is a registered JAX pytree (an equinox Module): its `key`,
    `bound_low`, and `bound_up` are dynamic array leaves. Construction stores one
    element's values unbatched, so a constraint is a single-element object; a
    collection stacks same-type constraints along a new leading axis and the
    optimizer `vmap`s the constraint over it. The key is resolved to an index
    against an equilibrium structure at evaluation time. Subclasses supply the
    constrained quantity via `constraint`. Missing bounds normalize to negative
    or positive infinity rather than None.

    A per-element constraint takes exactly one element key; to bound many
    elements, create one constraint per element and let collections vectorize
    them. Only aggregate constraints, which declare `is_aggregate = True`, span a
    whole structure.
    """

    key: Int[Array, "..."] = eqx.field(converter=as_key)  # pyright: ignore[reportAssignmentType]
    bound_low: Float[Array, "..."] = eqx.field(converter=as_bound_low, default=None)  # pyright: ignore[reportAssignmentType]
    bound_up: Float[Array, "..."] = eqx.field(converter=as_bound_up, default=None)  # pyright: ignore[reportAssignmentType]

    # An aggregate constraint spans a whole structure (all edges or nodes) in one
    # call: its key is the sentinel, its constraint reads the whole state in one
    # go, and it is never grouped with same-type peers into a vectorized
    # collection since it is already a batch of its own. A ClassVar so it stays a
    # plain class attribute a subclass overrides, out of the pytree.
    is_aggregate: ClassVar[bool] = False

    def keys_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "elements ..."]:
        """
        The structure's canonical keys for this constraint's vocabulary.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        keys_canonical :
            The node, edge, or vertex key ordering the constraint's key is
            resolved against, one entry per element: a node/vertex key, or an
            edge key pair.

        Notes
        -----
        The only per-subclass part of key resolution: a subclass names the
        canonical ordering its key belongs to (nodes, vertices, or edge pairs)
        and raises if the structure is the wrong kind. The resolution itself
        lives once in `index`, so an author who defines a constraint on a new
        vocabulary overrides this hook alone.
        """
        raise NotImplementedError

    def index(self, structure: EquilibriumStructure) -> Int[Array, "..."]:
        """
        The constraint's element index, resolved against the structure.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            A single index for a per-element constraint, or the whole index row
            for an aggregate constraint, which reads its whole selection in one
            call.

        Notes
        -----
        The real work of key resolution: a subclass names its canonical keys via
        `keys_from_structure`, and this resolves the constraint's key against them
        in one `indices_from_keys` call, never a per-key loop. The key stays a
        compile-time constant (a constraint is closed over, never a jitted
        argument), so the resolution folds to a constant at trace time. The key's
        own shape distinguishes a per-element constraint from an aggregate, so
        there is no aggregate special case here.
        """
        keys_canonical = self.keys_from_structure(structure)

        return indices_from_keys(keys_canonical, self.key)

    def __call__(
        self,
        eqstate: EquilibriumState,
        structure: EquilibriumStructure,
    ) -> Float[Array, "..."]:
        """
        Evaluate the constraint for one element against an equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to read the constrained quantity from.
        structure :
            The structure whose element ordering resolves the constraint's index.

        Returns
        -------
        constraint :
            The constrained quantity for this one element, in the raw shape its
            hook returns: a scalar for a per-element constraint, or the whole row
            for an aggregate.

        Raises
        ------
        ValueError
            If a per-element constraint's value is not a scalar, typically
            because the constraint was handed a list of keys where one was
            expected.

        Notes
        -----
        A constraint holds one element, so this is the linear per-element body:
        resolve the index, read the constrained quantity, and check its shape. A
        standalone call returns one element's unbatched value.

        To evaluate a group of same-type constraints, stack them into one pytree
        and `vmap`, exactly as one maps any equinox module over a batch::

            stacked = tree_stack(constraints)
            values = jax.vmap(lambda c: c(eqstate, structure))(stacked)

        The optimizer owns that `vmap`; a scalar constraint lacks a goal's
        prediction-vs-target shape check, so the per-element scalar guard here
        keeps a stray list key from silently mis-sizing against its bounds.
        """
        index = self.index(structure)
        value = self.constraint(eqstate, structure, index)

        if not self.is_aggregate and value.shape != ():
            raise ValueError(
                f"{type(self).__name__}: constraint value shape {value.shape} is "
                "not a scalar. A per-element constraint takes a single element "
                "key; pass one constraint per element instead of a list of keys.",
            )

        return value

    def evaluate(
        self,
        datastructure: FDNetwork | FDMesh,
        sparse: bool = True,
    ) -> Float[Array, "..."]:
        """
        Evaluate the constraint directly on a datastructure, without solving.

        Parameters
        ----------
        datastructure :
            The network or mesh to read the equilibrium state from. Its geometry
            is used as-is; no form-finding is run.
        sparse :
            If True, assemble the equilibrium state with the sparse model.

        Returns
        -------
        constraint :
            The constrained quantity for this element, in its raw shape.

        Notes
        -----
        A convenience for prototyping a constraint on the high-level COMPAS layer.
        Unlike the optimizer path, it consumes the datastructure's current state
        directly rather than solving for equilibrium from raw parameters.
        """
        equilibrium = datastructure_state(datastructure, sparse)

        return self(equilibrium.eq_state, equilibrium.structure)

    def constraint(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        Extract the constrained quantity for one element from an equilibrium state.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the quantity from.
        structure :
            The structure the constraint is evaluated against, so a constraint can
            read whole trace-time constants (connectivity, topology) from it. Most
            constraints ignore it.
        index :
            The element index, one for a per-element constraint or the whole index
            row for an aggregate constraint. A constraint carrying an extra
            per-element array reads it off `self`.

        Returns
        -------
        constraint :
            The value of the constrained quantity.
        """
        raise NotImplementedError
