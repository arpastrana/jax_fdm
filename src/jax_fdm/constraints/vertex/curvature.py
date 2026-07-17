from jax_fdm.constraints.node import NodeCurvatureConstraint
from jax_fdm.constraints.vertex import VertexConstraint
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumStructure


class VertexCurvatureConstraint(VertexConstraint, NodeCurvatureConstraint):
    """
    Bound the discrete curvature at a vertex over its one-hop neighborhood.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeCurvatureConstraint`: the
    constraint logic is inherited unchanged, while the constrained keys and
    their neighborhood polygon keys resolve against the vertices of a mesh.
    """

    def key_index(self, structure: EquilibriumStructure) -> dict[int, int]:
        """
        The key-to-index mapping used to resolve the neighborhood polygon keys.

        Parameters
        ----------
        structure :
            The mesh structure whose vertex ordering defines the indices.

        Returns
        -------
        key_index :
            The mapping from vertex keys to structure indices.

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

        return structure.vertex_index
