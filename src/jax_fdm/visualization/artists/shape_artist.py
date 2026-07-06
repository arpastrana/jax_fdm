from math import fabs

from compas.datastructures import Mesh
from compas.geometry import Cylinder
from compas.geometry import Line
from compas.geometry import Sphere
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from jax_fdm.visualization.shapes import Arrow

__all__ = ["FDShapeArtist"]


class FDShapeArtist:
    """
    A mixin that draws the elements of a force density datastructure as COMPAS shapes.

    It implements the geometry-producing hooks of :class:`FDDatastructureArtist`
    (spheres for points, cylinders for edges, arrow shapes for load and reaction
    vectors) with plain COMPAS geometry, so that any backend that can display
    COMPAS shapes and meshes can reuse them. The scene-pushing machinery is left
    to the backend artists.
    """
    arrow_bodywidth = 0.012
    arrow_headportion = 0.12
    arrow_headwidth = 0.04

    # ==========================================================================
    # Draw one element
    # ==========================================================================

    def draw_edge(self, edge, width, *args, **kwargs):
        """
        Draw an edge as a cylinder.
        """
        start, end = self.datastructure.edge_coordinates(edge)
        line = Line(start, end)

        return Cylinder.from_line_and_radius(line, width / 2.0)

    def draw_point(self, point, size, *args, **kwargs):
        """
        Draw a point as a sphere.
        """
        return Sphere(radius=size / 2.0, point=self._point_coordinates(point))

    def draw_load(self, point, scale, *args, **kwargs):
        """
        Draw a load vector at a point.
        """
        vector = self._point_load(point)

        if length_vector(vector) < self.load_tol:
            return

        xyz = self._point_coordinates(point)

        # shift start to make the arrow head touch the point the load is applied to
        start = add_vectors(xyz, scale_vector(vector, -scale))

        # shift start to account for half the size of the edge thickness
        widths = []
        for edge in self._point_edges(point):
            width = self.edge_width.get(edge)
            if not width:
                width = 0.0
            widths.append(width)

        start = add_vectors(start, scale_vector(normalize_vector(vector), -max(widths)))

        return self.draw_vector(vector, start, scale)

    def draw_reaction(self, point, scale, *args, **kwargs):
        """
        Draw a reaction vector at a point.
        """
        vector = self._point_reaction(point)
        start = self._point_coordinates(point)

        if length_vector(vector) < self.reaction_tol:
            return

        # shift the starting point if the max force of connected edges is compressive
        connected_edges = list(self._point_edges(point))
        if len(connected_edges) == 0:
            return

        forces = [self.datastructure.edge_force(e) for e in connected_edges]
        max_force = max(forces, key=lambda f: fabs(f))
        if max_force < 0.0:
            start = add_vectors(start, scale_vector(vector, scale))

        # reverse the vector to display the direction of the reaction forces
        return self.draw_vector(scale_vector(vector, -1.0), start, scale)

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def draw_vector(self, vector, start, scale):
        """
        Build an arrow shape from a vector.
        """
        vector_scaled = scale_vector(vector, scale)

        return Arrow(position=start,
                     direction=vector_scaled,
                     head_portion=self.arrow_headportion,
                     head_width=self.arrow_headwidth,
                     body_width=self.arrow_bodywidth)

    @staticmethod
    def arrow_to_mesh(arrow):
        """
        Convert an :class:`Arrow` shape into a mesh a scene can render.

        The scene backends have no registered scene object for our custom
        ``Arrow`` shape, but they render a :class:`compas.datastructures.Mesh`
        directly.
        """
        return Mesh.from_vertices_and_faces(*arrow.to_vertices_and_faces())
