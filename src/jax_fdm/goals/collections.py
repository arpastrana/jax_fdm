from itertools import groupby

from dfdm.goals import Goal


class Collection:
    pass


class GoalCollection(Collection):
    """
    A collection of goals of the same type to speed up optimization.
    """
    """
    def __new__(self, goals):
        # index -> indices
        when calling prediction(), target(), it should call vmapped versions of goal
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

        return gcls(keys=keys, weights=weights, targets=targets)


    # def __init__(self, goals):
        # TODO: check if goals are of the same type before going any further
        # self.type = None
        # self.goals = goals

    def index(self):
        """
        The indices of the network elements in the collection.

        Returns
        -------
        indices : Tuple[int]
            The indices of the elements.
        """
        pass

    def key(self):
        """
        The keys of the elements in the network.

        Returns
        -------
        keys : Tuple[int]
            The keys of the elements in network space.
        """
        pass

    def weight(self):
        """
        The importance of the goal.

        Returns
        -------
        weights : jnp.array (n, )
            An array with the weight assigned to each of the n goals.
        """
        pass

    def __call__(self, eqstate):
        """

        Returns
        -------
        gstate : GoalState
            The current goal state relative to an equilibrium state.
        """
        pass



if __name__ == "__main__":

    from jax_fdm.goals import EdgesLengthGoal
    from jax_fdm.goals import NodesPointGoal
    from collections import Counter


    goal_a = EdgesLengthGoal([(0, 1), (1, 2)], targets=[0.5, 0.75])
    goal_b = NodesPointGoal([3, 4, 5], targets=[[0., 0., 0.], [1., 1., 1.]])

    goal_a = EdgesLengthGoal((1, 2), 0.5)
    goal_b = NodesPointGoal(3, targets=[0., 0., 0.])
    goal_c = EdgesLengthGoal((0, 1), 1.5)


    goals = [goal_a, goal_b, goal_a, goal_b, goal_c]

    counter = Counter(goals)

    # sort goals by class name
    keyfunc = lambda g: type(g).__name__
    goals = sorted(goals, key=keyfunc)
    keyfunc = lambda g: type(g)
    groups = groupby(goals, keyfunc)

    goal_collections = []
    for key, goal_group in groups:
        gc = GoalCollection(list(goal_group))
        print(f"{type(gc)}:\n{gc._key}\n{gc._weight}\n{gc._target}\n\n")
        goal_collections.append(gc)

