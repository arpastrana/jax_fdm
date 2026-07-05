from compas.datastructures import Mesh
from compas.geometry import Cone
from compas.geometry import Cylinder
from compas.geometry import Line
from compas.geometry import Shape
from compas.geometry import Vector

__all__ = ["Arrow"]


class Arrow(Shape):
    """
    An arrow defined by the position where it starts and the direction vector
    it points to.

    Notes
    -----
    COMPAS core does not ship an arrow shape, and neither compas_viewer nor
    compas_notebook provide one, so we keep our own here. It is shared by every
    visualization backend that needs to draw load and reaction vectors.
    """
    def __init__(self,
                 position=[0, 0, 0],
                 direction=[0, 0, 1],
                 head_portion=0.3,
                 head_width=0.1,
                 body_width=0.02):
        super().__init__()
        self.position = Vector(*position)
        self.direction = Vector(*direction)
        self.head_portion = head_portion
        self.head_width = head_width
        self.body_width = body_width

    # ==========================================================================
    # Data
    # ==========================================================================

    @property
    def __data__(self):
        return {"position": list(self.position), "direction": list(self.direction)}

    @classmethod
    def __from_data__(cls, data):
        return cls(position=data["position"], direction=data["direction"])

    # ==========================================================================
    # Customization
    # ==========================================================================

    def __repr__(self):
        return "Arrow({0}, {1})".format(self.position, self.direction)

    # ==========================================================================
    # Methods
    # ==========================================================================

    def to_vertices_and_faces(self, u=8, **kwargs):
        """
        Returns a list of vertices and faces.

        Parameters
        ----------
        u : int, optional
            Number of faces in the "u" direction. Default is ``8``.

        Returns
        -------
        (vertices, faces)
            A list of vertex locations and a list of faces, with each face
            defined as a list of indices into the list of vertices.
        """
        if u < 3:
            raise ValueError("The value for u should be u > 3.")

        length = self.direction.length
        head_vector = self.direction * self.head_portion
        head_base = self.position + self.direction - head_vector

        # Head of the arrow (a cone from the head base up to the tip).
        # ``Cone.from_line_and_radius`` centers the base circle on the line
        # midpoint and puts the apex a full length beyond, so the axis line is
        # centered on ``head_base`` to seat the base there and the tip at the end.
        head_line = Line(head_base - head_vector * 0.5, head_base + head_vector * 0.5)
        cone = Cone.from_line_and_radius(head_line, self.head_width * length)
        v, f = cone.to_vertices_and_faces(u=u)
        head = Mesh.from_vertices_and_faces(v, f)

        # Body of the arrow (a cylinder from the start up to the head base).
        body_line = Line(self.position, head_base)
        cylinder = Cylinder.from_line_and_radius(body_line, self.body_width * length)
        v, f = cylinder.to_vertices_and_faces(u=u)
        body = Mesh.from_vertices_and_faces(v, f)

        body.join(head)

        return body.to_vertices_and_faces()

    def transform(self, transformation):
        """
        Transform the arrow.

        Parameters
        ----------
        transformation : :class:`compas.geometry.Transformation`
            The transformation used to transform the arrow.
        """
        self.position.transform(transformation)
        self.direction.transform(transformation)
