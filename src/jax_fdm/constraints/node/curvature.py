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
    Constraints the (discrete) curvature of a node based on its surrounding polygon of neighboring nodes.
    """
    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        polygon: Int[np.ndarray, "elements neighbors"],
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ):
        super().__init__(key, bound_low, bound_up)
        self.polygon = polygon
        self.index_polygon = None

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.index_polygon = self.polygon_indices(model, structure)

    def polygon_indices(self, model: EquilibriumModel, structure: EquilibriumStructure) -> Int[Array, "elements neighbors"]:
        """
        Obtains the indices of the polygon from a model.
        """
        index_max = max(self.index) + 1  # pyright: ignore[reportArgumentType]  # self.index is Optional by declaration but always populated by init() before this runs
        polygon = np.atleast_2d(self.polygon)
        index_polygon = np.zeros((index_max, polygon.shape[1]))
        for p, idx in zip(polygon, self.index):  # pyright: ignore[reportArgumentType]  # self.index is Optional by declaration but always populated by init() before this runs
            index_polygon[idx, :] = tuple([structure.node_index[nbr] for nbr in p])

        return jnp.array(index_polygon, dtype=jnp.int64)

    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the curvature at a node based on the xyz coordinates of its one-hop neighborhood.
        """
        point = eqstate.xyz[index, :]
        index_polygon = self.index_polygon[index, :]  # pyright: ignore[reportOptionalSubscript]  # self.index_polygon is Optional by declaration but always set in init() before this runs
        polygon = eqstate.xyz[index_polygon, :]

        return curvature_point_polygon(point, polygon)
