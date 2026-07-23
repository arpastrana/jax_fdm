from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm import DTYPE_INT_JAX
from jax_fdm.constraints.constraint import BoundLike
from jax_fdm.constraints.vertex.vertex import VertexConstraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import indices_from_keys
from jax_fdm.geometry import curvature_point_polygon

__all__ = ["VertexCurvatureConstraint"]


def _as_polygon(
    polygon: Int[Array, "neighbors"] | Sequence[int],
) -> Int[Array, "neighbors"]:
    """
    Coerce a neighbor ring to a JAX integer array.

    Parameters
    ----------
    polygon :
        The ordered neighboring vertex keys forming the ring around the vertex.

    Returns
    -------
    polygon :
        The neighbor keys as a JAX integer array, unbatched like the constraint's
        other leaves.
    """
    return jnp.asarray(polygon, dtype=DTYPE_INT_JAX)


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

    # kw_only lets this required leaf follow the base's defaulted bounds (the
    # dataclass "non-default after default" rule) without a default of its own.
    # The neighbor ring is stored as raw vertex keys, unbatched like the
    # constraint's other leaves, and resolved to structure indices inside
    # `constraint` where the query rides the evaluation vmap.
    polygon: Int[Array, "neighbors"] = eqx.field(kw_only=True)

    def __init__(
        self,
        key: int,
        polygon: Int[Array, "neighbors"] | Sequence[int],
        bound_low: BoundLike = None,
        bound_up: BoundLike = None,
    ) -> None:
        self.key = key
        self.polygon = _as_polygon(polygon)
        self.bound_low = bound_low
        self.bound_up = bound_up

    def constraint(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The discrete curvature at the vertex over its one-hop neighborhood.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
        structure :
            The mesh structure whose vertex ordering resolves the neighbor ring.
        index :
            The index of the vertex.

        Returns
        -------
        constraint :
            The discrete curvature at the vertex given its neighbors' coordinates.

        Notes
        -----
        The neighbor ring is stored as raw vertex keys, so it is resolved to
        structure indices here, inside the evaluation vmap, in a second
        `indices_from_keys` call whose ``(neighbors,)`` query rides the map. The
        canonical vertex ordering is static, so only the query search runs in
        JAX.
        """
        neighbors = indices_from_keys(structure.vertices, self.polygon)
        point = eq_state.xyz[index, :]
        polygon = eq_state.xyz[neighbors, :]

        return curvature_point_polygon(point, polygon)
