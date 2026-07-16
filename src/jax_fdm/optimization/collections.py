import inspect
from collections import defaultdict
from itertools import groupby
from typing import Any


class Collection:
    """
    A collection of goals of the same type to speed up optimization.
    """
    def __new__(cls, collectibles: list[Any]) -> Any:
        """
        Gather a collection of objects into a single vectorized object.
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
    Convert a list of goals into a list of vectorized goal collections.
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
    Convert a list of constraints into a list of vectorized constraint collections.
    """
    constraints = sorted(constraints, key=lambda g: type(g).__name__)
    groups = groupby(constraints, lambda g: type(g))

    collections = []
    for _, group in groups:
        collection = Collection(list(group))
        collections.append(collection)

    return collections
