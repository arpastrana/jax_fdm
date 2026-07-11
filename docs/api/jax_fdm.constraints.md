# Constraints

A constraint bounds a quantity of an equilibrium state between a lower and an
upper limit during constrained form-finding with
[`constrained_fdm`](jax_fdm.equilibrium.md).

## Base class

::: jax_fdm.constraints
    options:
      heading_level: 3
      members:
        - Constraint

---

## Edge constraints

::: jax_fdm.constraints.edge.length.EdgeLengthConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.edge.force.EdgeForceConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.edge.angle.EdgeAngleConstraint
    options:
      heading_level: 3

---

## Node constraints

::: jax_fdm.constraints.node.coordinates.NodeXCoordinateConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.node.coordinates.NodeYCoordinateConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.node.coordinates.NodeZCoordinateConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.node.curvature.NodeCurvatureConstraint
    options:
      heading_level: 3

---

## Vertex constraints

::: jax_fdm.constraints.vertex.coordinates.VertexXCoordinateConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.vertex.coordinates.VertexYCoordinateConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.vertex.coordinates.VertexZCoordinateConstraint
    options:
      heading_level: 3

---

## Network constraints

::: jax_fdm.constraints.network.length.NetworkEdgesLengthConstraint
    options:
      heading_level: 3

::: jax_fdm.constraints.network.force.NetworkEdgesForceConstraint
    options:
      heading_level: 3
