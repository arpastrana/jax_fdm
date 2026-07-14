from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel


class VertexConstraint(Constraint):
    """
    Base class for all constraints that pertain to a vertex in a mesh.
    """
    def index_from_model(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> int | tuple[int, ...]:
        """
        The index of the node in a structure.
        """
        try:
            return structure.vertex_index[self.key]  # pyright: ignore[reportArgumentType]  # self.key may be a single vertex key or a list of vertex keys; the dict lookup dispatches at runtime via the TypeError below
        except TypeError:
            return tuple([structure.vertex_index[k] for k in self.key])  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]  # self.key is Optional and may be a single int/tuple by declaration but always set to a list here, since a scalar key would not raise TypeError above
