# Custom constraints

When the constraint bank does not carry the limit your brief demands, you write your own.
A custom constraint is one class and one method away, the same deal as a [custom goal](custom_goals.md), and this guide assumes you have read [constraints](constraints.md) for the anatomy the recipe below builds on.
Because constraints reuse most of the goal machinery, there is little new to learn: mostly, we point out what changes.

## Recipe: A custom constraint

Suppose you want to *cap* how much any edge climbs, rather than nudge it toward a preferred rise.
A custom constraint subclasses an element family (`EdgeConstraint`, `NodeConstraint`, `VertexConstraint`) and implements `constraint`:

```python
import jax.numpy as jnp

from jax_fdm.constraints.edge import EdgeConstraint


class EdgeRiseConstraint(EdgeConstraint):
    """
    Bound the vertical rise of an edge.
    """

    def constraint(self, eq_state, index):
        vector = eq_state.vectors[index, :]

        return jnp.abs(vector[2])
```

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP


constraints = [EdgeRiseConstraint(edge, 0.0, 0.4) for edge in network.edges()]

network = constrained_fdm(
    network,
    optimizer=SLSQP(),
    loss=loss,
    parameters=parameters,
    constraints=constraints,
    maxiter=100,
)
```

The bounds arrive through the constructor as `bound_low` and `bound_up`, here pinning every rise between zero and 0.4.
The body of `constraint` is written for a single element and vectorized across all of them, exactly like a goal's `prediction` — and the [same purity rules](custom_goals.md#recipe-1-a-custom-scalar-goal) apply: `jnp` operations only, no Python branching on array values.

## Retargeting a constraint from nodes to vertices

The `NodeZCoordinateConstraint` from the [anatomy](constraints.md#the-anatomy-of-a-constraint) speaks only network: aim it at a mesh and you get a `TypeError`, because the two datastructures resolve keys through different index tables and silently mixing them is a subtle-bug factory.
The two dialects of [custom goals](custom_goals.md#recipe-4-retargeting-a-goal-from-nodes-to-vertices) rule constraints too: `Node*` constraints speak network, `Vertex*` constraints speak mesh, and crossing them points you at the right counterpart.

So how do you bound a mesh vertex's height?
You write the vertex twin of that class, and it is gloriously boring:

```python
from jax_fdm.constraints.vertex import VertexConstraint


class VertexZCoordinateConstraint(VertexConstraint, NodeZCoordinateConstraint):
    """
    Bound the Z coordinate of a vertex between a lower and an upper value.
    """
```

A docstring, an inheritance list with `VertexConstraint` first so the vertex key resolution wins the method resolution order, and nothing else, the `constraint` body comes along for the ride.
This one is real: it is the library's own `VertexZCoordinateConstraint`, verbatim, and every `Vertex*` constraint is built this exact way.

## Summary

One class, one method, and your constraint joins the differentiable machinery as a first-class citizen, honored by the `SLSQP` and `IPOPT` optimizers exactly like the built-in bounds.
And if you build a constraint that others might want, remember that contributions are warmly welcome 🚀.
