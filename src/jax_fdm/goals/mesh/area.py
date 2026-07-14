import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import area_polygon
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.mesh import MeshGoal


class MeshAreaGoal(ScalarGoal, MeshGoal):
    """
    Maximize the negative area of a mesh.
    """
    def __init__(self, target: float | Float[Array, "..."] = 0.0, weight: float = 1.0):
        super().__init__(key=-1, target=target, weight=weight)
        self.faces = None

    def init(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> None:
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        self.faces = structure.faces_indexed

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        The current load path of the network.
        """
        def face_xyz(face: Int[Array, "vertices"], xyz: Float[Array, "nodes 3"]) -> Float[Array, "vertices 3"]:
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
            xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return xyz_face

        def face_area(face: Int[Array, "vertices"], xyz: Float[Array, "nodes 3"]) -> Float[Array, "1"]:
            fxyz = face_xyz(face, xyz)
            return area_polygon(fxyz)

        faces_area = vmap(face_area, in_axes=(0, None))
        areas = faces_area(self.faces, eq_state.xyz)  # pyright: ignore[reportArgumentType]  # self.faces is Optional by declaration but always set in init() before this runs

        area = jnp.sum(areas) * -1.0

        return jnp.atleast_1d(area)


class MeshFacesAreaEqualizeGoal(ScalarGoal, MeshGoal):
    """
    Maximize the negative area of a mesh.
    """
    def __init__(self, target: float | Float[Array, "..."] = 0.0, weight: float = 1.0):
        super().__init__(key=-1, target=target, weight=weight)
        self.faces = None

    def init(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> None:
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        self.faces = structure.faces_indexed

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        The current load path of the network.
        """
        def face_xyz(face: Int[Array, "vertices"], xyz: Float[Array, "nodes 3"]) -> Float[Array, "vertices 3"]:
            """
            Get this face XYZ coordinates from XYZ vertices array.
            """
            face = jnp.ravel(face)

            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]
            xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return xyz_face

        def face_area(face: Int[Array, "vertices"], xyz: Float[Array, "nodes 3"]) -> Float[Array, "1"]:
            fxyz = face_xyz(face, xyz)
            return area_polygon(fxyz)

        faces_area = vmap(face_area, in_axes=(0, None))
        areas = faces_area(self.faces, eq_state.xyz)  # pyright: ignore[reportArgumentType]  # self.faces is Optional by declaration but always set in init() before this runs

        return jnp.atleast_1d(jnp.var(areas) / jnp.mean(areas))
