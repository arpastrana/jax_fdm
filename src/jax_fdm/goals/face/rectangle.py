import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import cosines_angles_polygon
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.face import FaceGoal


class FaceRectangularGoal(ScalarGoal, FaceGoal):
    """
    Make the internal angles of a quadrilateral mesh face reach 90 degrees.

    Notes
    -----
    This goal is only applicable to quadrilateral mesh faces.
    """

    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        weight: float = 1.0,
        target: float | Float[Array, "..."] = 0.0,
    ) -> None:
        super().__init__(key=key, target=target, weight=weight)
        # set in init() from the mesh structure, before any prediction runs
        self.face_indices: Int[Array, "faces 4"]

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> None:
        """
        Bind the goal to a mesh, caching the four corner indices of each face.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure whose face ordering defines the indices.
        """
        super().init(model, structure)
        face_indices = structure.faces_indexed[self.index]
        self.face_indices = face_indices[:, :4]

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        A measure of how far a face is from rectangular.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the face coordinates from.
        index :
            The index of the face.

        Returns
        -------
        prediction :
            The summed mean absolute cosine of the face's corner angles, zero when
            every corner is a right angle.
        """
        fxyz = eq_state.xyz[self.face_indices]
        face_cosines = vmap(cosines_angles_polygon, in_axes=(0))(fxyz)
        face_cosines = jnp.mean(jnp.abs(face_cosines), axis=-1)

        return jnp.atleast_1d(jnp.sum(face_cosines))
