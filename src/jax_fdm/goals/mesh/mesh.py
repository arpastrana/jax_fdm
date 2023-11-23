from jax_fdm.goals import Goal


class MeshGoal(Goal):
    """
    Base class for all goals that pertain to a subset of the nodes, faces and edges of a mesh.
    """
    def __init__(self, key=-1, target=0.0, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def index_from_model(self, model, structure):
        """
        The index of the goal key in a structure.
        """
        return -1
