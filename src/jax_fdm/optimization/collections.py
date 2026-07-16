import inspect
from collections import defaultdict
from itertools import groupby
from typing import Any


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
                attr = getattr(collectible, key)
                ckwargs[key].append(attr)

        collection = cls(**ckwargs)
        collection._iscollection = True

        return collection


def collect_goals(goals: list[Any]) -> list[Any]:
    """
    Group goals by type into vectorized collections.

    Parameters
    ----------
    goals :
        The goals to group.

    Returns
    -------
    collections :
        One collection per type of collectible goal, plus a singleton collection for
        each goal flagged as not collectible.
    """
    goals_collectable = []
    goals_uncollectable = []

    for goal in goals:
        if goal.is_collectible:
            goals_collectable.append(goal)
        else:
            goals_uncollectable.append(goal)

    collections = []

    if goals_collectable:
        goals_sorted = sorted(goals_collectable, key=lambda g: type(g).__name__)
        groups = groupby(goals_sorted, lambda g: type(g))

        for _, group in groups:
            collection = Collection(list(group))
            collections.append(collection)

    if goals_uncollectable:
        for goal in goals_uncollectable:
            collections.append(Collection([goal]))

    return collections


def collect_constraints(constraints: list[Any]) -> list[Any]:
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
