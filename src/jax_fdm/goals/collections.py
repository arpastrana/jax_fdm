from itertools import groupby

from dfdm.goals import Goal


class Collection:
    pass


class GoalCollection(Collection, Goal):
    """
    A collection of goals of the same type to speed up optimization.
    """
    def __new__(cls, goals):
        # check homogenity
        gtypes = [type(g) for g in goals]
        if len(set(gtypes)) != 1:
            raise TypeError("The input goals are not of the same type!")

        # get keys
        keys = [goal.key for goal in goals]

        # get weights
        weights = [goal.weight for goal in goals]

        # targets
        targets = [goal.target for goal in goals]

        # extract class
        gcls = gtypes.pop()

        return gcls(key=keys, target=targets, weight=weights)

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
