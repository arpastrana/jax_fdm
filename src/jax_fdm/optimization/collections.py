import inspect

from collections import defaultdict

from itertools import groupby


class Collection:
    """
    A collection of goals of the same type to speed up optimization.
    """
    def __new__(cls, collectibles):
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

        return cls(**ckwargs)


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    from jax_fdm.goals import EdgeLengthGoal
    from collections import Counter


    # goal_a = EdgesLengthGoal([(0, 1), (1, 2)], targets=[0.5, 0.75])
    # goal_b = NodesPointGoal([3, 4, 5], targets=[[0., 0., 0.], [1., 1., 1.]])

    # goal_b = NodesPointGoal(3, targets=[0., 0., 0.])
    goal_a = EdgeLengthGoal((1, 2), 0.5)
    goal_b = EdgeLengthGoal((0, 1), 1.5)

    goals = [goal_a, goal_b, goal_a]

    # counter = Counter(goals)

    # sort goals by class name
    keyfunc = lambda g: type(g).__name__
    goals = sorted(goals, key=keyfunc)
    keyfunc = lambda g: type(g)
    groups = groupby(goals, keyfunc)

    goal_collections = []
    for key, goal_group in groups:
        gc = GoalCollection(list(goal_group))
        print(f"Goal type{type(gc)}:\tKeys: {gc.key}\tWeights: {gc.weight.shape}{gc.weight}\tTargets: {gc.target.shape}{gc.target}\n\n")
        goal_collections.append(gc)
