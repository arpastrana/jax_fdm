from compas.geometry import Vector
from compas.geometry import Cone
from compas.geometry import Circle
from compas.geometry import Cylinder
from compas.geometry import Plane
from compas.geometry import Shape

from compas.datastructures import Mesh


__all__ = ["Arrow"]


class Arrow(Shape):
    """
    An arrow is defined by the position where it starts and direction vector where it points to.

    Notes
    -----
    This object is a verbatim copy of compas_view2.shapes.
    I paste it here because it is not available in COMPAS core, and
    we want to be able to visualize arrows in a notebook without having to
    install compas_view2.
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

    @property
    def data(self):
        """Returns the data dictionary that represents the Arrow.
        Returns
        -------
        dict
            The arrow data.
        """
        return {"position": list(self.position), "direction": list(self.direction)}

    @data.setter
    def data(self, data):
        self.position = Vector(*data["position"])
        self.direction = Vector(*data["direction"])

    # ==========================================================================
    # customization
    # ==========================================================================

    def __repr__(self):
        return "Arrow({0}, {1})".format(self.position, self.direction)

    # ==========================================================================
    # constructors
    # ==========================================================================

    @classmethod
    def from_data(cls, data):
        """Construct a Arrow from its data representation.

        Parameters
        ----------
        data : :obj:`dict`
            The data dictionary.
        Returns
        -------
        Arrow
            The constructed arrow.
        """
        arrow = cls()
        arrow.data = data
        return arrow

    # ==========================================================================
    # methods
    # ==========================================================================

    def to_vertices_and_faces(self, u=4):
        """Returns a list of vertices and faces.

        Parameters
        ----------
        u : int, optional
            Number of faces in the "u" direction.
            Default is ``4``.
        Returns
        -------
        (vertices, faces)
            A list of vertex locations and a list of faces,
            with each face defined as a list of indices into the list of vertices.
        """
        if u < 3:
            raise ValueError("The value for u should be u > 3.")

        # Head of arrow
        head_position = self.position + self.direction * (1 - self.head_portion)
        plane = Plane(head_position, self.direction)
        circle = Circle(plane, self.head_width * self.direction.length)
        cone = Cone(circle, self.direction.length * self.head_portion)
        v, f = cone.to_vertices_and_faces(u=u)
        head = Mesh.from_vertices_and_faces(v, f)

        # Body of arrow
        body_center = self.position + self.direction * (1 - self.head_portion) / 2
        plane = Plane(body_center, self.direction)
        circle = Circle(plane, self.body_width * self.direction.length)
        cylinder = Cylinder(circle, self.direction.length * (1 - self.head_portion))
        v, f = cylinder.to_vertices_and_faces(u=u)
        body = Mesh.from_vertices_and_faces(v, f)

        body.join(head)

        return body.to_vertices_and_faces()

    def transform(self, transformation):
        """Transform the Arrow.

        Parameters
        ----------
        transformation : :class:`Transformation`
            The transformation used to transform the cone.
        """
        self.position.transform(transformation)
        self.direction.transform(transformation)
