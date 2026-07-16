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
    An arrow shape defined by a start position and a direction vector.

    Parameters
    ----------
    position :
        The xyz coordinates where the arrow starts.
    direction :
        The direction vector the arrow points along; its length sets the arrow
        length.
    head_portion :
        The fraction of the arrow length taken up by the head.
    head_width :
        The head radius as a fraction of the arrow length.
    body_width :
        The body radius as a fraction of the arrow length.

    Notes
    -----
    COMPAS core does not ship an arrow shape, and neither compas_viewer nor
    compas_notebook provide one, so we keep our own here. It is shared by every
    visualization backend that needs to draw load and reaction vectors.
    """

    def __init__(
        self,
        position: list[float] = [0, 0, 0],
        direction: list[float] = [0, 0, 1],
        head_portion: float = 0.3,
        head_width: float = 0.1,
        body_width: float = 0.02,
    ) -> None:
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
        Compute the vertices of the arrow's discrete mesh.

        Returns
        -------
        vertices :
            The vertex coordinates of the tessellated arrow.
        """
        vertices, _ = self.to_vertices_and_faces()
        return vertices

    def compute_faces(self) -> list[Face]:
        """
        Compute the faces of the arrow's discrete mesh.

        Returns
        -------
        faces :
            The faces of the tessellated arrow, as lists of vertex indices.
        """
        _, faces = self.to_vertices_and_faces()
        return faces

    def to_vertices_and_faces(
        self,
        triangulated: bool = False,
        u: int | None = None,
        v: Any = None,
    ) -> tuple[list[Vertex], list[Face]]:
        """
        Tessellate the arrow into vertices and faces.

        Parameters
        ----------
        triangulated :
            If True, triangulate the faces.
        u :
            The number of faces around the arrow. Defaults to ``self.resolution_u``.
        v :
            Ignored; an arrow has no "v" direction.

        Returns
        -------
        vertices_and_faces :
            The vertex coordinates and the faces, each face a list of indices into
            the vertices.

        Raises
        ------
        ValueError
            If ``u`` is less than three.
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
        head_vertices, head_faces = cone.to_vertices_and_faces(
            u=u,
            triangulated=triangulated,
        )

        # Manually join the vertices and faces of the body and the head
        offset = len(vertices)
        joined_vertices: list[Vertex] = list(vertices) + list(head_vertices)
        joined_faces: list[Face] = list(faces)
        joined_faces += [[index + offset for index in face] for face in head_faces]

        return joined_vertices, joined_faces

    def transform(self, transformation: Transformation) -> None:
        """
        Transform the arrow in place.

        Parameters
        ----------
        transformation :
            The transformation applied to the arrow's position and direction.
        """
        self.position.transform(transformation)
        self.direction.transform(transformation)
