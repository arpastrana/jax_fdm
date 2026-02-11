from jax import vmap

import jax.numpy as jnp

from jax_fdm.geometry import planarity_triangle
from jax_fdm.geometry import planarity_polygon

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.mesh import MeshGoal


class MeshPlanarityGoal(ScalarGoal, MeshGoal):
    """
    Planarize the faces of a mesh.

    Notes
    -----
    This goal computes the average planarity of the faces of a mesh.
    The planarity of a face is calculated as the absolute dot product between
    the face's unitized normal vector and its unitized edge vectors.

    This function is experimental and it is unclear whether it works correctly
    for padded faces or faces with more than 4 vertices. Use with caution!
    """
    def __init__(self, target=0.0, weight=1.0):
        super().__init__(key=-1, target=target, weight=weight)
        self.faces_indexed = None

    def init(self, model, structure):
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        self.faces_indexed = structure.faces_indexed

    def prediction(self, eq_state, index):
        """
        The average planarity of the mesh faces.
        """
        planarities = faces_planarity(self.faces_indexed, eq_state.xyz)

        return jnp.atleast_1d(jnp.mean(planarities))


# ==========================================================================
# Planarity
# ==========================================================================

def face_xyz(face, xyz):
    """
    Get the xyz coordinates of a face from the xyz vertices array.

    Notes
    -----
    This function replaces -1 entries in the vertex indices with the index of the
    first vertex of the face. The negative indices are used for padding to end
    up with a uniform face array, otherwise, JAX complains.
    """
    face = jnp.ravel(face)
    xyz_face = xyz[face, :]
    xyz_repl = xyz_face[0, :]

    return vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)


def face_planarity(face, xyz):
    """
    Calculate face planarity according to the number of vertices in the face.

    Notes
    -----
    A triangle face has a planarity of 0.0 by construction.
    """
    valid_face_indices = jnp.where(face >= 0, 1.0, 0.0)
    sum_indices = jnp.sum(valid_face_indices)

    fxyz = face_xyz(face, xyz)
    planarity = jnp.where(sum_indices == 3, planarity_triangle(fxyz), planarity_polygon(fxyz))

    return planarity


def faces_planarity(faces, xyz):
    """
    Compute the planarity of a set of mesh faces.
    """
    return vmap(face_planarity, in_axes=(0, None))(faces, xyz)
