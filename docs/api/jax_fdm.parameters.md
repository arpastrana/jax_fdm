# jax_fdm.parameters

A parameter marks a quantity of the structure — a force density, a load
component, a support coordinate — as a variable the
[optimizer](jax_fdm.optimization.md) is allowed to change, optionally bounded.

## Base classes

::: jax_fdm.parameters
    options:
      heading_level: 3
      members:
        - Parameter
        - ParameterGroup
        - ParameterManager

---

## Edge parameters

::: jax_fdm.parameters
    options:
      heading_level: 3
      members:
        - EdgeForceDensityParameter
        - EdgeGroupForceDensityParameter

---

## Node parameters

::: jax_fdm.parameters
    options:
      heading_level: 3
      members:
        - NodeLoadXParameter
        - NodeLoadYParameter
        - NodeLoadZParameter
        - NodeSupportXParameter
        - NodeSupportYParameter
        - NodeSupportZParameter
        - NodeGroupLoadXParameter
        - NodeGroupLoadYParameter
        - NodeGroupLoadZParameter
        - NodeGroupSupportXParameter
        - NodeGroupSupportYParameter
        - NodeGroupSupportZParameter

---

## Vertex parameters

::: jax_fdm.parameters
    options:
      heading_level: 3
      members:
        - VertexLoadXParameter
        - VertexLoadYParameter
        - VertexLoadZParameter
        - VertexSupportXParameter
        - VertexSupportYParameter
        - VertexSupportZParameter
        - VertexGroupLoadXParameter
        - VertexGroupLoadYParameter
        - VertexGroupLoadZParameter
        - VertexGroupSupportXParameter
        - VertexGroupSupportYParameter
        - VertexGroupSupportZParameter

---

## Helpers

::: jax_fdm.parameters
    options:
      heading_level: 3
      members:
        - combine_parameters
        - split_parameters
        - reshape_parameters
