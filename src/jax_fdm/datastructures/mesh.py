"""
A catalogue of force density datastructures.
"""
from math import fabs

from compas.datastructures import Mesh
from compas.geometry import transform_points

from jax_fdm.datastructures import FDNetwork
from jax_fdm.datastructures import NodeMixins


class FDMesh(Mesh, NodeMixins):
# class FDMesh(FDNetwork, NodeMixins, Mesh):
    """
    A force density mesh.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_default_face_attributes({"px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0})
