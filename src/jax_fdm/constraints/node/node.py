from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class NodeConstraint(Constraint):
    """
    Base class for all constraints that pertain to a node in a network.
    """
    def index_from_model(self, model: EquilibriumModel, structure: EquilibriumStructure) -> int | tuple[int, ...]:
        """
        The index of the node in a structure.
        """
        try:
            return structure.node_index[self.key]  # pyright: ignore[reportArgumentType]  # self.key may be a single node key or a list of node keys; the dict lookup dispatches at runtime via the TypeError below
        except TypeError:
            return tuple([structure.node_index[k] for k in self.key])  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]  # self.key is Optional and may be a single int/tuple by declaration but always set to a list here, since a scalar key would not raise TypeError above
