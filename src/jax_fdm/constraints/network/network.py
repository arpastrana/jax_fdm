import equinox as eqx
from jaxtyping import Array
from jaxtyping import Int

from jax_fdm.constraints.constraint import Constraint
from jax_fdm.constraints.constraint import as_key
from jax_fdm.equilibrium import EquilibriumStructure

__all__ = ["NetworkConstraint"]


class NetworkConstraint(Constraint):
    """
    The base class for constraints defined on a network as a whole.

    Parameters
    ----------
    bound_low :
        The lower bound applied to every element. If None, unbounded below.
    bound_up :
        The upper bound applied to every element. If None, unbounded above.

    Notes
    -----
    A network constraint spans all edges or nodes at once rather than one element,
    so it is an aggregate that always carries the sentinel key ``-1``. The key is
    fixed with ``init=False``, so it is hidden from the constructor rather than
    exposed as a passable-but-ignored argument, and the whole network-constraint
    family constructs from bounds alone.
    """

    is_aggregate = True

    key: Int[Array, "..."] = eqx.field(converter=as_key, default=-1, init=False)  # pyright: ignore[reportAssignmentType]

    def index(
        self,
        structure: EquilibriumStructure,
    ) -> Int[Array, "..."]:
        """
        Return the sentinel index shared by all network constraints.

        Parameters
        ----------
        structure :
            The structure the constraint is bound to.

        Returns
        -------
        index :
            The sentinel index ``-1``, since the constraint spans the whole
            network and there is no element key to resolve.

        Notes
        -----
        A whole-structure aggregate has nothing to resolve, so it overrides
        `index` to short-circuit the key resolution the per-element constraints
        run, returning the sentinel unchanged. Every network-constraint reads the
        whole structure directly and ignores this index.
        """
        return self.key
