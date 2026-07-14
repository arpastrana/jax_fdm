import inspect
from collections import defaultdict
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
