from jax_fdm.goals import Goal


class FaceGoal(Goal):
    """
    Base class for all constraints that pertain to a face in a mesh.
    """
    def index_from_model(self, model, structure):
        """
        The index of the face in a structure.
        """
        try:
            return structure.face_index[self.key]
        except TypeError:
            return tuple([structure.face_index[k] for k in self.key])
