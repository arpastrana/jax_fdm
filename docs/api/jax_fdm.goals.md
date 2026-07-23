# Goals

A goal describes a target property of an equilibrium state — a coordinate, a
length, a force, an angle — that an optimizer should steer the structure toward.
Goals are collected into an error term of a [loss function](jax_fdm.losses.md).

## Base classes

::: jax_fdm.goals
    options:
      heading_level: 3
      members:
        - Goal
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

::: jax_fdm.goals.node.residual.NodeResidualPlaneGoal
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

Vertex goals are thin counterparts of the node goals above: the goal logic is
inherited unchanged, while keys resolve against the vertices of a mesh instead
of the nodes of a network. Applying one to a network raises a `TypeError`
pointing to the `Node*` counterpart, and vice versa. Only the goals with no
node equivalent — such as [VertexNormalAngleGoal][jax_fdm.goals.vertex.normal.VertexNormalAngleGoal]
and [VertexTangentAngleGoal][jax_fdm.goals.vertex.tangent.VertexTangentAngleGoal],
which need face topology — implement their own logic.

::: jax_fdm.goals.vertex.point.VertexPointGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.line.VertexLineGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.plane.VertexPlaneGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.segment.VertexSegmentGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.coordinates.VertexXCoordinateGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.coordinates.VertexYCoordinateGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.coordinates.VertexZCoordinateGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.residual.VertexResidualForceGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.residual.VertexResidualVectorGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.residual.VertexResidualDirectionGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.residual.VertexResidualPlaneGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.colinear.VerticesColinearGoal
    options:
      heading_level: 3

::: jax_fdm.goals.vertex.colinear.VerticesCurvatureGoal
    options:
      heading_level: 3

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
