from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.node.node import NodeConstraint
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import indices_from_keys
from jax_fdm.geometry import curvature_point_polygon

__all__ = ["NodeCurvatureConstraint"]


class NodeCurvatureConstraint(NodeConstraint):
    """
    Bound the discrete curvature at a node over its one-hop neighborhood.

    Parameters
    ----------
    key :
        The key of the node the constraint acts on.
    polygon :
        The neighboring node keys forming the polygon around the node.
    bound_low :
        The lower bound on the curvature. If None, unbounded below.
    bound_up :
        The upper bound on the curvature. If None, unbounded above.
    """

    def __init__(
        self,
        key: int,
        polygon: Int[Array, "nodes neighbors"] | Sequence[int],
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ) -> None:
        super().__init__(key, bound_low, bound_up)
        self.polygon = jnp.asarray(polygon)

    def keys_canonical(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "nodes"]:
        """
        The canonical node-key ordering the neighborhood polygon resolves against.

        Parameters
        ----------
        structure :
            The structure whose node ordering defines the indices.

        Returns
        -------
        keys :
            The structure's node keys, in canonical order.
        """
        return structure.nodes

    def operand(
        self,
        structure: EquilibriumStructure,
    ) -> tuple[Int[np.ndarray, "nodes"], Int[np.ndarray, "nodes neighbors"]]:
        """
        The per-element payload: the node indices paired with polygon neighbors.

        Parameters
        ----------
        structure :
            The structure whose node ordering defines the indices.

        Returns
        -------
        payload :
            The constrained-node index array and, per node, the structure indices
            of its neighborhood polygon, both collection ordered so vmap zips each
            node's index with its own neighbor row.
        """
        index = self.indices(structure)
        polygon = np.atleast_2d(np.asarray(self.polygon))
        neighbors = indices_from_keys(self.keys_canonical(structure), polygon.ravel())
        neighbors = neighbors.reshape(polygon.shape)

        return index, neighbors

    def constraint(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        payload: tuple[Int[Array, ""], Int[Array, "neighbors"]],
    ) -> Float[Array, ""]:
        """
        The discrete curvature at the node over its one-hop neighborhood.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the node coordinates from.
        structure :
            The structure the constraint is evaluated against; unused.
        payload :
            The node index and its polygon neighbor indices for this element.

        Returns
        -------
        constraint :
            The discrete curvature at the node given its neighbors' coordinates.
        """
        index, neighbors = payload
        point = eq_state.xyz[index, :]
        polygon = eq_state.xyz[neighbors, :]

        return curvature_point_polygon(point, polygon)
