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
    Constraints the (discrete) curvature of a node based on a polygon of neighboring nodes.
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
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.index_polygon = self.polygon_indices(model, structure)

    def polygon_indices(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        ) -> Int[Array, "nodes neighbors"]:
        """
        Obtains the indices of the polygon from a model.
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
        Returns the curvature at a node based on the xyz coordinates of its one-hop neighborhood.
        """
        point = eq_state.xyz[index, :]
        index_polygon = self.index_polygon[index, :]
        polygon = eq_state.xyz[index_polygon, :]

        return curvature_point_polygon(point, polygon)
