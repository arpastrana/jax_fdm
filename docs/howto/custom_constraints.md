# Custom constraints

A custom constraint is one class and one method away, the same in spirit as a [custom goal](custom_goals.md).
This guide assumes you have read [constraints](constraints.md) for the anatomy the recipe below builds on.
Because constraints reuse most of the goal machinery, there is little new to learn, and we mostly point out what changes.

## Recipe: A custom constraint

Suppose we want to *cap* the force magnitude in any edge, keeping it below a ceiling rather than nudging it toward a target as the [custom goal](custom_goals.md#recipe-1-a-custom-scalar-goal) did.
A custom constraint subclasses an element family (`EdgeConstraint`, `NodeConstraint`, `VertexConstraint`) and implements `constraint`:

```python
import jax.numpy as jnp

from jax_fdm.constraints.edge import EdgeConstraint


class EdgeForceMagnitudeConstraint(EdgeConstraint):
    """
    Bound the absolute internal force of an edge.
    """

    def constraint(self, eq_state, index):
        return jnp.abs(eq_state.forces[index, 0])
```

We hand a list of these constraints to `constrained_fdm` alongside the `loss` and `parameters` we set up as in [constrained form-finding](constrained_form_finding.md).
Next, let's utilize an optimizer that is able to ingest these constraints:

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP


constraints = [EdgeForceMagnitudeConstraint(edge, bound_low=0.0, bound_up=2.0) for edge in network.edges()]

network = constrained_fdm(
    network,
    optimizer=SLSQP(),
    loss=loss,
    parameters=parameters,
    constraints=constraints,
    maxiter=100,
)
```

The bounds arrive through the constructor as `bound_low` and `bound_up`, here keeping every edge force magnitude between zero and 2.0.
The body of `constraint` is written for a single element and vectorized by the numerical core across all of them, exactly like a goal's `prediction`.
The [same function purity rules](custom_goals.md#recipe-1-a-custom-scalar-goal) apply: `jnp` operations only, no Python branching or looping on array values.

## Retargeting a constraint from nodes to vertices

The `NodeZCoordinateConstraint` from the [anatomy](constraints.md#the-anatomy-of-a-constraint) speaks only network: aim it at a mesh and we get a `TypeError`, because the two datastructures resolve keys through different index tables - silently mixing them is a subtle-bug factory.
The two dialects of [custom goals](custom_goals.md#recipe-4-retargeting-a-goal-from-nodes-to-vertices) rule constraints too: `Node*` constraints speak network, `Vertex*` constraints speak mesh, and crossing them points us at the right counterpart.

So how do we bound a mesh vertex's height?
We simply write the vertex twin of that class:

```python
from jax_fdm.constraints.vertex import VertexConstraint


class VertexZCoordinateConstraint(VertexConstraint, NodeZCoordinateConstraint):
    """
    Bound the Z coordinate of a vertex between a lower and an upper value.
    """
```

The only requirements are a docstring and an inheritance list with `VertexConstraint` first, so the vertex key resolution wins the method resolution order.
The `constraint` body comes along for the ride, as well as all the other computation machinery inherited from the parent classes.
The constraint above is real though: it is the library's own `VertexZCoordinateConstraint`, verbatim, and every `Vertex*` constraint is built this exact way.

## Summary

Customized constraints can join the differentiable machinery of JAX FDM as a first-class citizen, honored by the `SLSQP` and `IPOPT` optimizers exactly like the built-in bounds.
Remember to contribute if you think your future self might find your custom constraint useful ;)
