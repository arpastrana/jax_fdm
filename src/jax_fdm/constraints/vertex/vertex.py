from jax_fdm.constraints import Constraint


class VertexConstraint(Constraint):
    """
    Base class for all constraints that pertain to a vertex in a mesh.
    """
    def index_from_model(self, model, structure):
        """
        The index of the node in a structure.
        """
        try:
            return structure.vertex_index[self.key]
        except TypeError:
            return tuple([structure.vertex_index[k] for k in self.key])
