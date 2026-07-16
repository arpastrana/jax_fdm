from typing import Any

from compas.geometry import Cone
from compas.geometry import Cylinder
from compas.geometry import Line
from compas.geometry import Shape
from compas.geometry import Transformation
from compas.geometry import Vector

__all__ = ["Arrow"]

# A discrete mesh vertex is an xyz triple; a face is a list of vertex indices,
# or a 3-tuple of indices once the shape is triangulated.
Vertex = list[float]
Face = list[int] | tuple[int, int, int]


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
                 position: list[float] = [0, 0, 0],
                 direction: list[float] = [0, 0, 1],
                 head_portion: float = 0.3,
                 head_width: float = 0.1,
                 body_width: float = 0.02) -> None:
        super().__init__()
        self.position = Vector(*position)
        self.direction = Vector(*direction)
        self.head_portion = head_portion
        self.head_width = head_width
        self.body_width = body_width
        self.resolution_u = 8

    # ==========================================================================
    # Data
    # ==========================================================================

    @property
    def __data__(self) -> dict[str, list[float]]:
        return {"position": list(self.position), "direction": list(self.direction)}

    @classmethod
    def __from_data__(cls, data: dict[str, list[float]]) -> "Arrow":
        return cls(position=data["position"], direction=data["direction"])

    # ==========================================================================
    # Customization
    # ==========================================================================

    def __repr__(self) -> str:
        return "Arrow({0}, {1})".format(self.position, self.direction)

    # ==========================================================================
    # Methods
    # ==========================================================================

    def compute_vertices(self) -> list[Vertex]:
        """
        Compute the vertices of the discrete representation of the arrow.
        """
        vertices, _ = self.to_vertices_and_faces()
        return vertices

    def compute_faces(self) -> list[Face]:
        """
        Compute the faces of the discrete representation of the arrow.
        """
        _, faces = self.to_vertices_and_faces()
        return faces

    def to_vertices_and_faces(self, triangulated: bool = False, u: int | None = None, v: Any = None) -> tuple[list[Vertex], list[Face]]:
        """
        Returns a list of vertices and faces.

        Parameters
        ----------
        triangulated : bool, optional
            If ``True``, triangulate the faces.
        u : int, optional
            Number of faces in the "u" direction.
            Defaults to ``self.resolution_u``.
        v : int, optional
            Ignored. An arrow has no "v" direction.

        Returns
        -------
        (vertices, faces)
            A list of vertex locations and a list of faces, with each face
            defined as a list of indices into the list of vertices.
        """
        u = u or self.resolution_u
        if u < 3:
            raise ValueError("The value for u should be u > 3.")

        length = self.direction.length
        head_vector = self.direction * self.head_portion
        head_base = self.position + self.direction - head_vector

        # Body of the arrow (a cylinder from the start up to the head base)
        body_line = Line(self.position, head_base)
        cylinder = Cylinder.from_line_and_radius(body_line, self.body_width * length)
        vertices, faces = cylinder.to_vertices_and_faces(u=u, triangulated=triangulated)

        # Head of the arrow (a cone from the head base up to the tip)
        head_line = Line(head_base - head_vector * 0.5, head_base + head_vector * 0.5)
        cone = Cone.from_line_and_radius(head_line, self.head_width * length)
        head_vertices, head_faces = cone.to_vertices_and_faces(u=u, triangulated=triangulated)

        # Manually join the vertices and faces of the body and the head
        offset = len(vertices)
        joined_vertices: list[Vertex] = list(vertices) + list(head_vertices)
        joined_faces: list[Face] = list(faces)
        joined_faces += [[index + offset for index in face] for face in head_faces]

        return joined_vertices, joined_faces

    def transform(self, transformation: Transformation) -> None:
        """
        Transform the arrow.

        Parameters
        ----------
        transformation : :class:`compas.geometry.Transformation`
            The transformation used to transform the arrow.
        """
        self.position.transform(transformation)
        self.direction.transform(transformation)
