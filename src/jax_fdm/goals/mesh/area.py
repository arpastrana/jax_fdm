import jax.numpy as jnp

from jax import vmap

from jax_fdm.goals import ScalarGoal

from jax_fdm.goals.mesh import MeshGoal

from jax_fdm.geometry import area_polygon


class MeshAreaGoal(ScalarGoal, MeshGoal):
    """
    Maximize the negative area of a mesh.
    """
    def __init__(self, target=0.0, weight=1.0):
        super().__init__(key=-1, target=target, weight=weight)
        self.faces = None

    def init(self, model, structure):
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        self.faces = structure.faces_indexed

    def prediction(self, eq_state, index):
        """
        The current load path of the network.
        """
        def face_xyz(face, xyz):
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

        def face_area(face, xyz):
            fxyz = face_xyz(face, xyz)
            return area_polygon(fxyz)

        faces_area = vmap(face_area, in_axes=(0, None))
        areas = faces_area(self.faces, eq_state.xyz)

        area = jnp.sum(areas) * -1.0

        return jnp.atleast_1d(area)


class MeshFacesAreaEqualizeGoal(ScalarGoal, MeshGoal):
    """
    Maximize the negative area of a mesh.
    """
    def __init__(self, target=0.0, weight=1.0):
        super().__init__(key=-1, target=target, weight=weight)
        self.faces = None

    def init(self, model, structure):
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        self.faces = structure.faces_indexed

    def prediction(self, eq_state, index):
        """
        The current load path of the network.
        """
        def face_xyz(face, xyz):
            """
            Get this face XYZ coordinates from XYZ vertices array.
            """
            face = jnp.ravel(face)

            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]
            xyz_face = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return xyz_face

        def face_area(face, xyz):
            fxyz = face_xyz(face, xyz)
            return area_polygon(fxyz)

        faces_area = vmap(face_area, in_axes=(0, None))
        areas = faces_area(self.faces, eq_state.xyz)

        return jnp.atleast_1d(jnp.var(areas) / jnp.mean(areas))
