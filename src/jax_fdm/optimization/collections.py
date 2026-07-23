from collections.abc import Sequence
from itertools import groupby
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.tree_util import tree_map

if TYPE_CHECKING:
    from jax_fdm.constraints import Constraint
    from jax_fdm.goals import Goal

__all__ = ["collect_goals", "collect_constraints"]


def tree_stack(collectibles: Sequence["Goal | Constraint"]) -> "Goal | Constraint":
    """
    Stack same-type goals or constraints into one vectorized pytree.

    Parameters
    ----------
    collectibles :
        The goals or constraints to vectorize; all must share one type.

    Returns
    -------
    collection :
        A single object of the shared type whose dynamic leaves carry a leading
        element axis holding the whole group, so it evaluates in one vmapped
        call.

    Notes
    -----
    A goal's or constraint's key is a dynamic array leaf, not a static field, so
    same-type members differing only in their key share one pytree structure and
    `tree_map` stacks them leaf by leaf. Each member stores one element's values
    unbatched, so `jnp.stack` adds a new leading axis of the group size with no
    frozen-module escape hatch. Stacking one member (`tree_stack([c])`) still adds
    the axis, so a lone goal or constraint becomes a collection of one.
    """
    return tree_map(lambda *leaves: jnp.stack(leaves), *collectibles)


def collect_goals(goals: Sequence["Goal"]) -> list["Goal"]:
    """
    Group goals by type into vectorized collections.

    Parameters
    ----------
    goals :
        The goals to group.

    Returns
    -------
    collections :
        One collection per type of per-element goal, plus a singleton collection
        for each aggregate goal, which is already a batch of its own.

    Notes
    -----
    A collection is built by `tree_stack`, which stacks the goals' array leaves
    into one vectorized goal evaluated in a single vmapped call. An aggregate goal
    reduces over its elements on its own, so each becomes a batch of one.
    """
    goals_element = []
    goals_aggregate = []

    for goal in goals:
        if goal.is_aggregate:
            goals_aggregate.append(goal)
        else:
            goals_element.append(goal)

    collections = []

    if goals_element:
        goals_sorted = sorted(goals_element, key=lambda g: type(g).__name__)
        groups = groupby(goals_sorted, lambda g: type(g))

        for _, group in groups:
            collection = tree_stack(list(group))
            collections.append(collection)

    for goal in goals_aggregate:
        collections.append(tree_stack([goal]))

    return collections


def collect_constraints(constraints: Sequence["Constraint"]) -> list["Constraint"]:
    """
    Group constraints by type into vectorized collections.

    Parameters
    ----------
    constraints :
        The constraints to group.

    Returns
    -------
    collections :
        One collection per type of per-element constraint, plus a singleton
        collection for each aggregate constraint, which is already a batch of its
        own.

    Notes
    -----
    A collection is built by `tree_stack`, which stacks the constraints' array
    leaves into one vectorized constraint evaluated in a single vmapped call. An
    aggregate constraint spans its whole structure on its own, so each becomes a
    batch of one.
    """
    constraints_element = []
    constraints_aggregate = []

    for constraint in constraints:
        if constraint.is_aggregate:
            constraints_aggregate.append(constraint)
        else:
            constraints_element.append(constraint)

    collections = []

    if constraints_element:
        constraints_sorted = sorted(
            constraints_element,
            key=lambda c: type(c).__name__,
        )
        groups = groupby(constraints_sorted, lambda c: type(c))

        for _, group in groups:
            collection = tree_stack(list(group))
            collections.append(collection)

    for constraint in constraints_aggregate:
        collections.append(tree_stack([constraint]))

    return collections
