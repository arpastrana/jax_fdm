from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class EdgeConstraint(Constraint):
    """
    Base class for all constraints that pertain to an edge of a network.
    """
    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        ) -> int | tuple[int, ...]:
        """
        The index of the edge key in an equilibrium structure.
        """
        return self._index_from_key(structure.edge_index)
