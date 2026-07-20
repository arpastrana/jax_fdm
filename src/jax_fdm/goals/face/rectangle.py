import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import cosines_angles_polygon
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.face import FaceGoal
from jax_fdm.goals.goal import TargetLike


class FaceRectangularGoal(ScalarGoal, FaceGoal):
    """
    Make the internal angles of a quadrilateral mesh face reach 90 degrees.

    Notes
    -----
    This goal is only applicable to quadrilateral mesh faces.
    """

    def __init__(
        self,
        key: int,
        weight: float = 1.0,
        target: TargetLike = 0.0,
    ) -> None:
        super().__init__(key=key, target=target, weight=weight)
        # set in init() from the mesh structure, before any prediction runs
        self.faces_indexed: Int[Array, "faces vertices"]

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> None:
        """
        Bind the goal to a mesh, caching the face topology.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure whose face ordering defines the indices.
        """
        super().init(model, structure)
        self.faces_indexed = structure.faces_indexed

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
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
            The mean absolute cosine of the face's corner angles, zero when
            every corner is a right angle.
        """
        fxyz = eq_state.xyz[self.faces_indexed[index, :4]]
        face_cosines = cosines_angles_polygon(fxyz)

        return jnp.mean(jnp.abs(face_cosines))
