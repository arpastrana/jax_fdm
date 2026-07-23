import inspect
from collections import defaultdict
from collections.abc import Sequence
from itertools import groupby
from typing import TYPE_CHECKING
from typing import Any

import jax.numpy as jnp
from jax.tree_util import tree_map

if TYPE_CHECKING:
    from jax_fdm.constraints import Constraint
    from jax_fdm.goals import Goal

__all__ = ["Collection", "collect_goals", "collect_constraints"]


def tree_stack(goals: Sequence["Goal"]) -> "Goal":
    """
    Stack same-type goals into one vectorized goal by stacking their leaves.

    Parameters
    ----------
    goals :
        The goals to vectorize; all must share one type.

    Returns
    -------
    collection :
        A single goal of the shared type whose dynamic leaves carry a leading
        element axis holding the whole group, so it evaluates in one vmapped
        call.

    Notes
    -----
    A goal's key is a dynamic array leaf, not a static field, so same-type goals
    differing only in their key share one pytree structure and `tree_map` stacks
    them leaf by leaf. Each goal stores one element's values unbatched, so
    `jnp.stack` adds a new leading axis of the group size with no frozen-module
    escape hatch. Stacking one goal (`tree_stack([g])`) still adds the axis, so a
    lone goal becomes a collection of one.
    """
    return tree_map(lambda *leaves: jnp.stack(leaves), *goals)


class Collection:
    """
    A vectorized stand-in for many same-type goals or constraints.

    Notes
    -----
    Instantiating a Collection returns a single object of the collectibles' own
    class whose init arguments are the stacked per-object values, so the whole group
    evaluates in one vmapped call. A ``_iscollection`` flag marks the result.
    """

    def __new__(cls, collectibles: list[Any]) -> Any:
        """
        Stack same-type objects into one vectorized object of their class.

        Parameters
        ----------
        collectibles :
            The objects to vectorize; all must share one type.

        Returns
        -------
        collection :
            A single object of the shared class, built from the per-object init
            arguments gathered across the group.

        Raises
        ------
        TypeError
            If the collectibles are not all of the same type.
        AttributeError
            If an init parameter is not stored as a same-named attribute.
        """
        # check homogenity
        ctypes = [type(c) for c in collectibles]
        if len(set(ctypes)) != 1:
            raise TypeError("The input collectibles are not of the same type!")

        # extract class
        cls = ctypes.pop()

        # class signature
        sig = inspect.signature(cls)

        # collect init signature values
        ckwargs = defaultdict(list)
        for key in sig.parameters.keys():
            for collectible in collectibles:
                try:
                    attr = getattr(collectible, key)
                except AttributeError as error:
                    raise AttributeError(
                        f"{cls.__name__}.__init__ parameter '{key}' must be "
                        f"stored as attribute 'self.{key}': collections rebuild "
                        "goals and constraints from their init signature.",
                    ) from error
                ckwargs[key].append(attr)

        # Flag the instance before __init__ runs, so the key setter can tell a
        # collection rebuild (stacked keys are legitimate) from a user passing
        # a key list to a per-element goal or constraint.
        collection = object.__new__(cls)
        collection._iscollection = True
        collection.__init__(**ckwargs)

        return collection


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
        One collection per type of constraint.
    """
    constraints = sorted(constraints, key=lambda g: type(g).__name__)
    groups = groupby(constraints, lambda g: type(g))

    collections = []
    for _, group in groups:
        collection = Collection(list(group))
        collections.append(collection)

    return collections
