# Custom constraints

Goals express *soft* preference: get as close to the target as you can.
Constraints express *hard* restrictions: stay between these bounds, or the optimizer does not go home.

This guide assumes you have read [Custom goals](goals.md).
Constraints reuse most of that machinery, so here we focus on what changes.

## How a constraint works

A constraint lives the same two-phase life as a goal.
You construct it with an element key and two bounds, and when you call `constrained_fdm`, its `init` resolves the key to an integer index inside the equilibrium structure.
Keys at construction, array rows at evaluation, `init` in between — the [same translation step](goals.md#how-a-goal-works) goals go through.

The differences are three.

- **Bounds instead of a target.**
A constraint carries no target and no weight; it carries `bound_low` and `bound_up`, and the optimizer keeps the constrained quantity between them.
Leave either bound as `None` and it becomes an infinity of the appropriate sign, so one-sided constraints cost you nothing.
- **Law needs an enforcer.**
Constraints are honored only by optimizers that support them, `SLSQP` and `IPOPT`.
Hand constraints to any other optimizer and they are politely ignored.
- **One method, different name.**
Where a goal implements `prediction`, a constraint implements `constraint`.
Same signature, same job: slice the quantity of interest for one element out of an `EquilibriumState`.

One constraint, one key, same as goals.
To bound many elements, create one constraint per element, and the machinery batches same-type constraints into a single vectorized call.

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
The body of `constraint` is written for a single element and vectorized across all of them, exactly like a goal's `prediction` — and the [same purity rules](goals.md#recipe-1-a-custom-scalar-goal) apply: `jnp` operations only, no Python branching on array values.

## Retargeting a constraint from nodes to vertices

The two dialects of [Custom goals](goals.md#recipe-4-retargeting-a-goal-from-nodes-to-vertices) rule constraints too: `Node*` constraints speak network, `Vertex*` constraints speak mesh, and crossing them raises a `TypeError` pointing at the right counterpart.
The remedy is the same thin twin:

```python
from jax_fdm.constraints.vertex import VertexConstraint


class VertexZCoordinateConstraint(VertexConstraint, NodeZCoordinateConstraint):
    """
    Bound the Z coordinate of a vertex between a lower and an upper value.
    """
```

A docstring, an inheritance list with `VertexConstraint` first, and nothing else.
This one is real: it is the library's own `VertexZCoordinateConstraint`, verbatim.
