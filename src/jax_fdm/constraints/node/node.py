from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class NodeConstraint(Constraint):
    """
    Base class for all constraints that pertain to a node in a network.
    """
    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        ) -> int | tuple[int, ...]:
        """
        The index of the node in a structure.
        """
        return self._index_from_key(structure.node_index)
