import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import datastructure_state
from jax_fdm.equilibrium.indexing import _indices_from_keys

__all__ = ["Constraint"]


class Constraint:
    """
    The base class for all constraints, bounds an equilibrium quantity must obey.

    Parameters
    ----------
    key :
        The key of the element the constraint acts on.
    bound_low :
        The lower bound on the constrained quantity. If None, unbounded below.
    bound_up :
        The upper bound on the constrained quantity. If None, unbounded above.

    Notes
    -----
    A constraint is stateless: construction stores the key and bounds, and
    `__call__` resolves the key to an index against an equilibrium structure at
    evaluation time. Subclasses supply the constrained quantity via `constraint`.
    Missing bounds normalize to negative or positive infinity rather than None.

    A constraint takes exactly one element key; to bound many elements, create
    one constraint per element and let collections vectorize them.
    """

    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        bound_low: float | Float[Array, "..."] | None = None,
        bound_up: float | Float[Array, "..."] | None = None,
    ) -> None:
        self._key: int | tuple[int, int] | list[int] | list[tuple[int, int]] | None = (
            None
        )
        self.key = key

        # normalized to a finite bound or +/- inf by the setters; never None after
        self._bound_low: float | Float[Array, "..."]
        self.bound_low = bound_low

        self._bound_up: float | Float[Array, "..."]
        self.bound_up = bound_up

    @property
    def key(self) -> int | tuple[int, int] | list[int] | list[tuple[int, int]] | None:
        """
        The key of an element in a network.
        """
        return self._key

    @key.setter
    def key(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
    ) -> None:
        if isinstance(key, list):
            if not key:
                raise ValueError(
                    f"{type(self).__name__} got an empty key list. "
                    "Pass at least one element key.",
                )
            if not getattr(self, "_iscollection", False):
                raise TypeError(
                    f"{type(self).__name__} takes a single element key, got a "
                    f"list of {len(key)}. Create one constraint per element "
                    "(collections vectorize same-type constraints "
                    "automatically).",
                )
        self._key = key

    @staticmethod
    def _bound_setter(
        bound: float | Float[Array, "..."],
    ) -> float | Float[Array, "..."]:
        """
        Normalize a bound to a scalar float or a flat array.

        Parameters
        ----------
        bound :
            The bound to normalize.

        Returns
        -------
        bound :
            The bound as a scalar float when it holds a single value, otherwise a
            flattened array.
        """
        if isinstance(bound, (int, float)):
            return bound
        bound = jnp.ravel(jnp.asarray(bound))
        if bound.size == 1:
            return float(bound[0])
        return bound

    @property
    def bound_low(self) -> float | Float[Array, "..."]:
        """
        The lower bound of this constraint.
        """
        return self._bound_low

    @bound_low.setter
    def bound_low(self, bound: float | Float[Array, "..."] | None) -> None:
        if bound is None:
            bound = -jnp.inf
        self._bound_low = self._bound_setter(bound)

    @property
    def bound_up(self) -> float | Float[Array, "..."]:
        """
        The upper bound of this constraint.
        """
        return self._bound_up

    @bound_up.setter
    def bound_up(self, bound: float | Float[Array, "..."] | None) -> None:
        if bound is None:
            bound = jnp.inf
        self._bound_up = self._bound_setter(bound)

    def index_from_structure(
        self,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Resolve the constraint's key to an index in an equilibrium structure.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            The index, or tuple of indices, of the constraint's element(s).
        """
        raise NotImplementedError

    def _indices_from_keys(
        self,
        keys_canonical: Int[np.ndarray, "elements ..."],
    ) -> int | tuple[int, ...]:
        """
        Resolve the constraint's key(s) to positions in a canonical key ordering.

        Parameters
        ----------
        keys_canonical :
            The structure's canonical key ordering (nodes, vertices, or edge
            pairs), one row per element.

        Returns
        -------
        index :
            A single index when the key resolves to one element, or a tuple of
            indices when it resolves to several.

        Notes
        -----
        The scalar-vs-tuple choice follows the number of resolved elements, not
        the Python type of the key, so a list, tuple, or array of several keys
        all yield a tuple. A single edge key ``(u, v)`` resolves to one element
        and stays a scalar.
        """
        key = self.key
        if key is None:
            raise ValueError(f"{type(self).__name__} has no key to resolve.")

        resolved = _indices_from_keys(keys_canonical, key)
        if len(resolved) == 1:
            return int(resolved[0])

        return tuple(int(index) for index in resolved)

    def indices(self, structure: EquilibriumStructure) -> Int[np.ndarray, "elements"]:
        """
        The constraint's element index as a one-dimensional array, ready for vmap.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        index :
            One entry per element, mapped by vmap to a per-element scalar.
        """
        return np.atleast_1d(np.asarray(self.index_from_structure(structure)))

    def operand(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "elements"] | tuple[Array, ...]:
        """
        The per-element payload vmap maps at axis 0.

        Parameters
        ----------
        structure :
            The structure whose element ordering defines the index.

        Returns
        -------
        payload :
            The index array by default. A constraint carrying an extra
            per-element array overrides this to return a tuple, which
            `constraint` unpacks. Every leaf is collection-ordered, so vmap zips
            row k of each into call k.
        """
        return self.indices(structure)

    def __call__(
        self,
        params: EquilibriumParametersState,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "elements"]:
        """
        Evaluate the constraint by solving for equilibrium first.

        Parameters
        ----------
        params :
            The parameters defining the equilibrium problem.
        model :
            The equilibrium model that computes the equilibrium state.
        structure :
            The structure that provides the connectivity.

        Returns
        -------
        constraint :
            The constrained quantity for each element, flattened.
        """
        eqstate = model(params, structure)

        return self._constraint(eqstate, structure)

    def _constraint(
        self,
        eqstate: EquilibriumState,
        structure: EquilibriumStructure,
    ) -> Float[Array, "elements"]:
        """
        Evaluate the constraint on a precomputed equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to read the constrained quantity from.
        structure :
            The structure whose element ordering resolves the constraint's index.

        Returns
        -------
        constraint :
            The constrained quantity for each element, flattened.

        Notes
        -----
        The shared core for both a solving `__call__` and a state-consuming
        `evaluate`. One per-element payload is mapped at axis 0; eqstate and
        structure are held out (in_axes=None), so structure rides whole and
        sparse matrices never become vmap operands.
        """
        payload = self.operand(structure)
        constraint = vmap(self.constraint, in_axes=(None, None, 0))(
            eqstate,
            structure,
            payload,  # pyright: ignore[reportArgumentType]
        )

        return jnp.ravel(constraint)

    def evaluate(
        self,
        datastructure: FDNetwork | FDMesh,
        sparse: bool = True,
    ) -> Float[Array, "elements"]:
        """
        Evaluate the constraint directly on a datastructure, without solving.

        Parameters
        ----------
        datastructure :
            The network or mesh to read the equilibrium state from. Its geometry
            is used as-is; no form-finding is run.
        sparse :
            If True, assemble the equilibrium state with the sparse model.

        Returns
        -------
        constraint :
            The constrained quantity for each element, flattened.

        Notes
        -----
        A convenience for prototyping a constraint on the high-level COMPAS layer.
        Unlike `__call__`, it consumes the datastructure's current state directly
        rather than solving for equilibrium from raw parameters.
        """
        equilibrium = datastructure_state(datastructure, sparse)

        return self._constraint(equilibrium.eq_state, equilibrium.structure)

    def constraint(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        payload: Int[Array, ""] | tuple[Array, ...],
    ) -> Float[Array, "..."]:
        """
        Extract the constrained quantity for one element from an equilibrium state.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the quantity from.
        structure :
            The structure the constraint is evaluated against, held out of the
            vmap so a constraint can read whole trace-time constants from it.
            Most constraints ignore it.
        payload :
            The per-element slice vmap maps at axis 0. The index of the element
            for a plain constraint, or a tuple the constraint unpacks when it
            carries an extra per-element array (see `operand`).

        Returns
        -------
        constraint :
            The value of the constrained quantity.
        """
        raise NotImplementedError
