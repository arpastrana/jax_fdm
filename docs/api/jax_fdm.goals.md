# jax_fdm.goals

A goal describes a target property of an equilibrium state — a coordinate, a
length, a force, an angle — that an optimizer should steer the structure toward.
Goals are collected into an error term of a [loss function](jax_fdm.losses.md).

## Base classes

::: jax_fdm.goals
    options:
      heading_level: 3
      members:
        - Goal
        - ScalarGoal
        - VectorGoal
        - GoalState

---

## Node goals

::: jax_fdm.goals.node.point.NodePointGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.line.NodeLineGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.plane.NodePlaneGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.segment.NodeSegmentGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.coordinates.NodeXCoordinateGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.coordinates.NodeYCoordinateGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.coordinates.NodeZCoordinateGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.residual.NodeResidualForceGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.residual.NodeResidualVectorGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.residual.NodeResidualDirectionGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.colinear.NodesColinearGoal
    options:
      heading_level: 3

::: jax_fdm.goals.node.colinear.NodesCurvatureGoal
    options:
      heading_level: 3

---

## Edge goals

::: jax_fdm.goals.edge.length.EdgeLengthGoal
    options:
      heading_level: 3

::: jax_fdm.goals.edge.length.EdgesLengthEqualGoal
    options:
      heading_level: 3

::: jax_fdm.goals.edge.force.EdgeForceGoal
    options:
      heading_level: 3

::: jax_fdm.goals.edge.force.EdgesForceEqualGoal
    options:
      heading_level: 3

::: jax_fdm.goals.edge.direction.EdgeDirectionGoal
    options:
      heading_level: 3

::: jax_fdm.goals.edge.angle.EdgeAngleGoal
    options:
      heading_level: 3

::: jax_fdm.goals.edge.loadpath.EdgeLoadPathGoal
    options:
      heading_level: 3

---

## Vertex goals

::: jax_fdm.goals.vertex.normal.VertexNormalAngleGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.tangent.VertexTangentAngleGoal
    options:
      heading_level: 3

---

## Face goals

::: jax_fdm.goals.face.rectangle.FaceRectangularGoal
    options:
      heading_level: 3

---

## Mesh goals

::: jax_fdm.goals.mesh.area.MeshAreaGoal
    options:
      heading_level: 3

::: jax_fdm.goals.mesh.area.MeshFacesAreaEqualizeGoal
    options:
      heading_level: 3

::: jax_fdm.goals.mesh.laplacian.MeshXYZLaplacianGoal
    options:
      heading_level: 3

::: jax_fdm.goals.mesh.laplacian.MeshXYZFaceLaplacianGoal
    options:
      heading_level: 3

::: jax_fdm.goals.mesh.loadpath.MeshLoadPathGoal
    options:
      heading_level: 3

::: jax_fdm.goals.mesh.smoothing.MeshSmoothGoal
    options:
      heading_level: 3

::: jax_fdm.goals.mesh.planarity.MeshPlanarityGoal
    options:
      heading_level: 3

---

## Network goals

::: jax_fdm.goals.network.loadpath.NetworkLoadPathGoal
    options:
      heading_level: 3

::: jax_fdm.goals.network.laplacian.NetworkXYZLaplacianGoal
    options:
      heading_level: 3

::: jax_fdm.goals.network.smoothing.NetworkSmoothGoal
    options:
      heading_level: 3
