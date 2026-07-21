import numpy as np
from jaxtyping import Int

from jax_fdm.constraints.node.curvature import NodeCurvatureConstraint
from jax_fdm.constraints.vertex.vertex import VertexConstraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure

__all__ = ["VertexCurvatureConstraint"]


class VertexCurvatureConstraint(VertexConstraint, NodeCurvatureConstraint):
    """
    Bound the discrete curvature at a vertex over its one-hop neighborhood.

    Notes
    -----
    Both the constrained keys and their neighborhood polygon keys resolve
    against the vertices of a mesh.
    """

    def keys_canonical(
        self,
        structure: EquilibriumStructure,
    ) -> Int[np.ndarray, "vertices"]:
        """
        The canonical vertex-key ordering the neighborhood polygon resolves against.

        Parameters
        ----------
        structure :
            The mesh structure whose vertex ordering defines the indices.

        Returns
        -------
        keys :
            The mesh's vertex keys, in canonical order.

        Raises
        ------
        TypeError
            If the structure is not a mesh structure.
        """
        if not isinstance(structure, EquilibriumMeshStructure):
            raise TypeError(
                f"{type(self).__name__} targets mesh vertices. "
                "Use its Node* counterpart on a network.",
            )

        return structure.vertices
