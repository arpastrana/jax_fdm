from jax_fdm.constraints import Constraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


class EdgeConstraint(Constraint):
    """
    The base class for constraints defined on the edges of a network.
    """

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the constraint's edge key to an index in a structure.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose edge ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the constraint's edge(s).
        """
        return self._index_from_key(structure.edge_index)
