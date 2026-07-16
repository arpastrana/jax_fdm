from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure


class Constraint:
    """
    Base class for all constraints.
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

        # set in init() from the equilibrium structure, before any constraint runs
        self._index: Int[np.ndarray, "elements"]

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
        self._key = key

    @property
    def index(self) -> Int[np.ndarray, "elements"]:
        """
        The index of the constraint key in the canonical ordering of an
        equilibrium structure.
        """
        return self._index

    @index.setter
    def index(self, index: int | tuple[int, ...] | Int[np.ndarray, "elements"]) -> None:
        if isinstance(index, int):
            index = (index,)
        self._index = np.asarray(index)

    @staticmethod
    def _bound_setter(
        bound: float | Float[Array, "..."],
    ) -> float | Float[Array, "..."]:
        """
        Normalize a bound to a scalar float or a flat jax array.
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

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        Get the index in the model of the constraint key.
        """
        raise NotImplementedError

    def _index_from_key(self, key_index: dict[Any, int]) -> int | tuple[int, ...]:
        """
        Look up the constraint key in an element-to-index mapping.

        A single key maps to one index, while a list of keys maps to a tuple of indices.
        """
        if isinstance(self.key, list):
            return tuple(key_index[k] for k in self.key)
        return key_index[self.key]

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> None:
        """
        Initialize the constraint with information from an equilibrium model.
        """
        self.index = self.index_from_model(model, structure)

    def __call__(
        self,
        params: EquilibriumParametersState,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, "elements"]:
        """
        The called constraint function.
        """
        eqstate = model(params, structure)
        # self.index is a numpy index array mapped to a jax scalar by vmap
        constraint = vmap(self.constraint, in_axes=(None, 0))(eqstate, self.index)  # pyright: ignore[reportArgumentType]

        return jnp.ravel(constraint)

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        The constraint function.
        """
        raise NotImplementedError
