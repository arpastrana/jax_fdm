import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import area_polygon
from jax_fdm.goals.mesh.mesh import MeshGoal

__all__ = ["MeshAreaGoal", "MeshFacesAreaEqualizeGoal"]


class MeshAreaGoal(MeshGoal):
    """
    Drive the total surface area of a mesh toward a target.

    Notes
    -----
    The prediction is the negated total area, so a target of zero paired with a
    minimizing loss maximizes the mesh area.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The negated total surface area of the mesh.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        structure :
            The mesh structure providing the face indices.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The negated sum of the face areas.
        """

        def face_xyz(
            face: Int[Array, "vertices"],
            xyz: Float[Array, "vertices 3"],
        ) -> Float[Array, "vertices 3"]:
            """
            Get this face XYZ coordinates from XYZ vertices array.
            """
            face = jnp.ravel(face)

            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]

            # NOTE: Replace -1 with first entry to avoid nans in gradient computation
            # This was a pesky bug, since using nans as replacement did not cause
            # issues with the forward computation of normals, but it does for
            # the backward pass.
            xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(
                face >= 0,
                xyz_face,
                xyz_repl,
            )

            return xyz_face

        def face_area(
            face: Int[Array, "vertices"],
            xyz: Float[Array, "vertices 3"],
        ) -> Float[Array, ""]:
            fxyz = face_xyz(face, xyz)
            return area_polygon(fxyz)

        faces_area = vmap(face_area, in_axes=(0, None))
        areas = faces_area(structure.faces_indexed, eq_state.xyz)

        return jnp.sum(areas) * -1.0


class MeshFacesAreaEqualizeGoal(MeshGoal):
    """
    Equalize the areas of the faces of a mesh.

    Notes
    -----
    The goal drives the mean-normalized variance of the face areas to zero.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The variance of the face areas, normalized by their mean.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        structure :
            The mesh structure providing the face indices.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean-normalized variance of the face areas, zero when all equal.
        """

        def face_xyz(
            face: Int[Array, "vertices"],
            xyz: Float[Array, "vertices 3"],
        ) -> Float[Array, "vertices 3"]:
            """
            Get this face XYZ coordinates from XYZ vertices array.
            """
            face = jnp.ravel(face)

            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]
            xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(
                face >= 0,
                xyz_face,
                xyz_repl,
            )

            return xyz_face

        def face_area(
            face: Int[Array, "vertices"],
            xyz: Float[Array, "vertices 3"],
        ) -> Float[Array, ""]:
            fxyz = face_xyz(face, xyz)
            return area_polygon(fxyz)

        faces_area = vmap(face_area, in_axes=(0, None))
        areas = faces_area(structure.faces_indexed, eq_state.xyz)

        return jnp.var(areas) / jnp.mean(areas)
