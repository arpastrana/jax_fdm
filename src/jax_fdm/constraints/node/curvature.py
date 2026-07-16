import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.node import NodeConstraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.geometry import curvature_point_polygon


class NodeCurvatureConstraint(NodeConstraint):
    """
    Bound the discrete curvature at a node over its one-hop neighborhood.

    Parameters
    ----------
    key :
        The key or keys of the node(s) the constraint acts on.
    polygon :
        The neighboring node keys forming the polygon around each node.
    bound_low :
        The lower bound on the curvature. If None, unbounded below.
    bound_up :
        The upper bound on the curvature. If None, unbounded above.
    """

    def __init__(
        self,
        key: int | list[int],
        polygon: Int[Array, "nodes neighbors"],
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ) -> None:
        super().__init__(key, bound_low, bound_up)
        self.polygon = jnp.asarray(polygon)
        # set in init() from the equilibrium structure, before any constraint runs
        self.index_polygon: Int[Array, "nodes neighbors"]

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Bind the constraint to a structure, resolving its neighborhood polygon.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose node ordering defines the indices.
        """
        super().init(model, structure)
        self.index_polygon = self.polygon_indices(model, structure)

    def polygon_indices(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Int[Array, "nodes neighbors"]:
        """
        Resolve each node's neighborhood polygon keys to structure indices.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose node ordering defines the indices.

        Returns
        -------
        index_polygon :
            The neighbor indices of each constrained node, keyed by structure index.
        """
        index_max = max(self.index) + 1
        polygon = np.atleast_2d(self.polygon)
        index_polygon = np.zeros((index_max, polygon.shape[1]))
        for p, idx in zip(polygon, self.index):
            index_polygon[idx, :] = tuple([structure.node_index[nbr] for nbr in p])

        return jnp.array(index_polygon, dtype=jnp.int64)

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The discrete curvature at the node over its one-hop neighborhood.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the node coordinates from.
        index :
            The index of the node.

        Returns
        -------
        constraint :
            The discrete curvature at the node given its neighbors' coordinates.
        """
        point = eq_state.xyz[index, :]
        index_polygon = self.index_polygon[index, :]
        polygon = eq_state.xyz[index_polygon, :]

        return curvature_point_polygon(point, polygon)
