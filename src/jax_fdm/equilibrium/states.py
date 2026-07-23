from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium.structures import EquilibriumStructure

# ==========================================================================
# Equilibrium state
# ==========================================================================

__all__ = [
    "DatastructureState",
    "EquilibriumParametersState",
    "EquilibriumState",
    "LoadState",
]


class EquilibriumState(NamedTuple):
    """
    The static equilibrium of a structure under load.

    Attributes
    ----------
    xyz :
        The coordinates of the nodes.
    residuals :
        The residual force at each node, zero at nodes in equilibrium.
    lengths :
        The length of each edge.
    forces :
        The axial force in each edge.
    loads :
        The load applied to each node.
    vectors :
        The edge vectors pointing from tail to head node.
    """

    xyz: Float[Array, "nodes 3"]
    residuals: Float[Array, "nodes 3"]
    lengths: Float[Array, "edges 1"]
    forces: Float[Array, "edges 1"]
    loads: Float[Array, "nodes 3"]
    vectors: Float[Array, "edges 3"]

    @classmethod
    def from_datastructure(
        cls,
        datastructure: FDNetwork | FDMesh,
        structure: EquilibriumStructure,
        dtype: DTypeLike | None = None,
    ) -> "EquilibriumState":
        """
        Read an equilibrium state off a datastructure's stored attributes.

        Parameters
        ----------
        datastructure :
            The network or mesh to read coordinates, residuals, lengths, forces,
            and loads from.
        structure :
            The structure whose connectivity derives the edge vectors, the one
            state field the datastructure does not store.
        dtype :
            The floating-point dtype for the state arrays. If None, uses
            ``DTYPE_JAX``.

        Returns
        -------
        eq_state :
            The equilibrium state mirroring the datastructure's attributes.

        Raises
        ------
        ValueError
            If the datastructure is neither a network nor a mesh.

        Notes
        -----
        The state is read as-is, without form-finding: every field but the edge
        vectors comes straight from a stored attribute, so it is only a genuine
        equilibrium once the datastructure has been solved. Run ``fdm`` (or
        ``constrained_fdm``) on the datastructure first; on a fresh, never-solved
        input the lengths, forces, and residuals default to zero. The edge
        vectors are derived as ``connectivity @ xyz`` since the datastructure
        never persists them.
        """
        if dtype is None:
            dtype = DTYPE_JAX

        if isinstance(datastructure, FDNetwork):
            xyz = datastructure.nodes_coordinates()
            residuals = datastructure.nodes_residual()
            loads = datastructure.nodes_loads()
        elif isinstance(datastructure, FDMesh):
            xyz = datastructure.vertices_coordinates()
            residuals = datastructure.vertices_residual()
            loads = datastructure.vertices_loads()
        else:
            raise ValueError(f"Input datastructure {datastructure} is invalid")

        xyz = jnp.asarray(xyz, dtype)
        vectors = structure.connectivity @ xyz

        lengths = jnp.asarray(datastructure.edges_lengths(), dtype)
        lengths = jnp.reshape(lengths, (-1, 1))

        forces = jnp.asarray(datastructure.edges_forces(), dtype)
        forces = jnp.reshape(forces, (-1, 1))

        return cls(
            xyz=xyz,
            residuals=jnp.asarray(residuals, dtype),
            lengths=lengths,
            forces=forces,
            loads=jnp.asarray(loads, dtype),
            vectors=vectors,
        )


# ==========================================================================
# Load state
# ==========================================================================


class LoadState(NamedTuple):
    """
    The loads applied to a structure, grouped by the element they act on.

    Attributes
    ----------
    nodes :
        The load applied directly to each node.
    edges :
        The line load on each edge, or ``0.0`` when there is no edge load.
    faces :
        The area load on each face, or ``0.0`` when there is no face load.

    Notes
    -----
    Edge and face loads collapse to the scalar ``0.0`` when all their entries
    vanish, which lets the model skip distributing them to the nodes. Networks
    always carry ``faces=0.0``.
    """

    nodes: Float[Array, "nodes 3"]
    edges: Float[Array, "edges 3"] | float
    faces: Float[Array, "faces 3"] | float

    @classmethod
    def from_datastructure(
        cls,
        datastructure: FDNetwork | FDMesh,
        dtype: DTypeLike | None = None,
    ) -> "LoadState":
        """
        Create a load state from a network or mesh.

        Parameters
        ----------
        datastructure :
            The network or mesh to read the node, edge, and face loads from.
        dtype :
            The floating-point dtype for the load arrays. If None, uses the array
            library default.

        Returns
        -------
        load_state :
            The load state, with edge and face loads collapsed to ``0.0`` when
            they are all zero.
        """
        loads_edges = jnp.asarray(datastructure.edges_loads(), dtype)
        if jnp.allclose(loads_edges, 0.0):
            loads_edges = 0.0

        if isinstance(datastructure, FDNetwork):
            loads_nodes = jnp.asarray(datastructure.nodes_loads(), dtype)
            loads_faces = 0.0

        elif isinstance(datastructure, FDMesh):
            loads_nodes = jnp.asarray(datastructure.vertices_loads(), dtype)

            loads_faces = jnp.asarray(datastructure.faces_loads(), dtype)
            if jnp.allclose(loads_faces, 0.0):
                loads_faces = 0.0

        return cls(nodes=loads_nodes, edges=loads_edges, faces=loads_faces)


# ==========================================================================
# Parameter state
# ==========================================================================


class EquilibriumParametersState(NamedTuple):
    """
    The independent parameters that define an equilibrium problem.

    Attributes
    ----------
    q :
        The force density of each edge.
    xyz_fixed :
        The coordinates of the fixed (supported) nodes.
    loads :
        The load state applied to the structure.
    """

    q: Float[Array, "edges"]
    xyz_fixed: Float[Array, "nodes_fixed 3"]
    loads: LoadState

    @classmethod
    def from_datastructure(
        cls,
        datastructure: FDNetwork | FDMesh,
        dtype: DTypeLike | None = None,
    ) -> "EquilibriumParametersState":
        """
        Create a parameter state from a network or mesh.

        Parameters
        ----------
        datastructure :
            The network or mesh to read force densities, fixed coordinates, and
            loads from.
        dtype :
            The floating-point dtype for the parameter arrays. If None, uses
            ``DTYPE_JAX``.

        Returns
        -------
        params_state :
            The force densities, fixed node coordinates, and load state.

        Raises
        ------
        ValueError
            If the datastructure is neither a network nor a mesh.
        """
        q = datastructure.edges_forcedensities()

        if isinstance(datastructure, FDNetwork):
            xyz_fixed = datastructure.nodes_fixedcoordinates()
        elif isinstance(datastructure, FDMesh):
            xyz_fixed = datastructure.vertices_fixedcoordinates()
        else:
            raise ValueError(f"Input datastructure {datastructure} is invalid")

        if dtype is None:
            dtype = DTYPE_JAX

        return cls(
            q=jnp.asarray(q, dtype),
            xyz_fixed=jnp.asarray(xyz_fixed, dtype),
            loads=LoadState.from_datastructure(datastructure),
        )


# ==========================================================================
# State bundled from a datastructure
# ==========================================================================


class DatastructureState(NamedTuple):
    """
    The equilibrium of a datastructure and the objects read off it.

    Attributes
    ----------
    eq_state :
        The equilibrium state read off the datastructure as-is, without solving.
    structure :
        The structure carrying the connectivity the state was assembled on.
    parameters :
        The force densities, fixed coordinates, and loads read off the
        datastructure.
    """

    eq_state: EquilibriumState
    structure: EquilibriumStructure
    parameters: EquilibriumParametersState
