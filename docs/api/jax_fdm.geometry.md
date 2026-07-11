# Geometry

Differentiable geometric primitives written in JAX, used by the
[goals](jax_fdm.goals.md) and [constraints](jax_fdm.constraints.md) to measure
an equilibrium state.

## Vectors

::: jax_fdm.geometry
    options:
      heading_level: 3
      members:
        - length_vector
        - length_vector_sqrd
        - normalize_vector
        - vector_unitized
        - subtract_vectors
        - vector_projection
        - cosine_vectors
        - angle_vectors
        - line_vector

---

## Points and distances

::: jax_fdm.geometry
    options:
      heading_level: 3
      members:
        - distance_point_point_sqrd
        - closest_point_on_line
        - closest_point_on_segment
        - closest_point_on_plane
        - colinearity_points
        - curvature_points
        - curvature_point_polygon

---

## Polygons

::: jax_fdm.geometry
    options:
      heading_level: 3
      members:
        - area_polygon
        - area_triangle
        - normal_polygon
        - normal_triangle
        - angles_polygon
        - cosines_angles_polygon
        - planarity_polygon
        - planarity_triangle

---

## Local coordinate systems

::: jax_fdm.geometry
    options:
      heading_level: 3
      members:
        - line_lcs
        - polygon_lcs
