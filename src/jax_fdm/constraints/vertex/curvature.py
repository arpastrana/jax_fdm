from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.vertex.vertex import VertexConstraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import indices_from_keys
from jax_fdm.geometry import curvature_point_polygon

__all__ = ["VertexCurvatureConstraint"]


class VertexCurvatureConstraint(VertexConstraint):
    """
    Bound the discrete curvature at a vertex over its one-hop neighborhood.

    Parameters
    ----------
    key :
        The key of the vertex the constraint acts on.
    polygon :
        The neighboring vertex keys forming the ordered ring around the vertex.
    bound_low :
        The lower bound on the curvature. If None, unbounded below.
    bound_up :
        The upper bound on the curvature. If None, unbounded above.

    Notes
    -----
    The discrete curvature is the angle deficit `2 * pi - sum(alphas)`, summed
    over the angles between successive spokes to the neighbor ring, so it is
    only meaningful when the neighbors are given in cyclic order. A mesh
    supplies that order canonically through its face winding (e.g. from
    `compas.datastructures.Mesh.vertex_neighbors(vkey, ordered=True)`); a plain
    network has no faces and hence no canonical ring, which is why the
    constraint is mesh only.
    """

    def __init__(
        self,
        key: int,
        polygon: Int[Array, "vertices neighbors"] | Sequence[int],
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ) -> None:
        super().__init__(key, bound_low, bound_up)
        self.polygon = jnp.asarray(polygon)

    def operand(
        self,
        structure: EquilibriumMeshStructure,
    ) -> tuple[Int[np.ndarray, "vertices"], Int[np.ndarray, "vertices neighbors"]]:
        """
        The per-element payload: the vertex indices paired with polygon neighbors.

        Parameters
        ----------
        structure :
            The mesh structure whose vertex ordering defines the indices.

        Returns
        -------
        payload :
            The constrained-vertex index array and, per vertex, the structure
            indices of its neighborhood polygon, both collection ordered so vmap
            zips each vertex's index with its own neighbor row.
        """
        index = self.indices(structure)
        polygon = np.atleast_2d(np.asarray(self.polygon))
        neighbors = indices_from_keys(structure.vertices, polygon.ravel())
        neighbors = neighbors.reshape(polygon.shape)

        return index, neighbors

    def constraint(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        payload: tuple[Int[Array, ""], Int[Array, "neighbors"]],
    ) -> Float[Array, ""]:
        """
        The discrete curvature at the vertex over its one-hop neighborhood.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
        structure :
            The structure the constraint is evaluated against; unused.
        payload :
            The vertex index and its polygon neighbor indices for this element.

        Returns
        -------
        constraint :
            The discrete curvature at the vertex given its neighbors' coordinates.
        """
        index, neighbors = payload
        point = eq_state.xyz[index, :]
        polygon = eq_state.xyz[neighbors, :]

        return curvature_point_polygon(point, polygon)
