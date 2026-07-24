# Constraints

Goals express *soft* preference: get as close to the target as we can.
Constraints express *hard* restrictions: stay between these bounds, or the optimizer does not go home.

This guide is about what a constraint *is* and how it differs from a goal.
It assumes we have read [goals](goals.md). Constraints reuse most of that machinery, so here we focus on what changes.
Once the anatomy is clear, [custom constraints](custom_constraints.md) shows how to write our own, and for how goals and constraints slot into a full optimization problem, see [constrained form-finding](constrained_form_finding.md).

## The anatomy of a constraint

We take a constraint apart the same top-down way the [goals](goals.md#the-anatomy-of-a-goal) guide takes apart `EdgeLengthGoal`.
Suppose that instead of *nudging* a quantity toward a target, we want a hard floor and ceiling: no node in the network may sink below zero or rise above three.
We grab `NodeZCoordinateConstraint`, one per node, and hand each a pair of bounds:

```python
from jax_fdm.constraints import NodeZCoordinateConstraint


constraints = [NodeZCoordinateConstraint(node, bound_low=0.0, bound_up=3.0) for node in network.nodes()]
```

One constraint per node, each pinning that node's height between 0.0 and 3.0.
The most evident difference with respect to a goal is already visible: where the goal took a `target`, the constraint takes two bounds.

### What we pass in

Every constraint is built from a key and two bounds, defined once as fields on the base `Constraint` and inherited by every constraint in the library:

```python
import equinox as eqx


class Constraint(eqx.Module):

    key: Array        # which element (resolved to an index at evaluation time)
    bound_low: Array  # lower limit; None becomes -inf
    bound_up: Array   # upper limit; None becomes +inf
```

Like a goal, a constraint is an [equinox](https://docs.kidger.site/equinox/) module: its fields are the leaves of a registered pytree, the constructor is synthesized, and a `None` bound is normalized to the matching infinity as it is stored. So, as with goals, the constraints we write declare no `__init__` of their own.
The `key` behaves exactly as a goal's: stored as given, the node key here, then resolved to an integer `index` against the structure when the optimizer evaluates the constraint.
But **the target and weight are gone**.
They are replaced by two bounds:

- **`bound_low`** and **`bound_up`** are the limits the optimizer must keep the quantity between. If we pass `None` for either, the constraint will convert it to the matching infinity, so a one-sided constraint (a floor with no ceiling, say) is straightforward to set up.
- There is **no weight**, because a constraint is not weighed against anything. It is satisfied or the solution is rejected, full stop.

### What is under the hood

Even less than a goal. Here is the whole class:

```python
from jax_fdm.constraints.node import NodeConstraint


class NodeZCoordinateConstraint(NodeConstraint):
    """
    Bound the Z coordinate of a node between a lower and an upper value.
    """

    def constraint(self, eq_state, structure, index):
        return eq_state.xyz[index, 2]
```

Two moving parts, mirroring the goal with two deliberate differences:

- **One base class, and no rank at all.** A constraint subclasses only an element family (`NodeConstraint`). There is nothing like a goal's scalar-versus-vector distinction to worry about: a constraint always reduces to a single number per element, so its `constraint` returns one scalar and `__call__` checks that it did.
- **The `constraint` method is the one thing we write.** It plays the exact role `prediction` plays for a goal, same signature, same job: given an `eq_state`, the `structure`, and the `index` our key resolved to, return the quantity of interest, here the node's Z coordinate pulled from the `xyz` array. Only the name differs.

So a constraint is a goal with the target-and-weight swapped for bounds and a single-family base class.
The next section makes those differences precise.

!!! note "Where the equilibrium state comes from"

    A goal and a constraint both take a ready-made `eq_state`: their `__call__(eq_state, structure)` slices the quantity out of it and never solves for equilibrium itself. The difference is *who* does the solving. A goal is evaluated inside the loss, on the state the loss already has in hand. A constraint is evaluated by the optimizer on its own schedule, apart from the loss: the optimizer solves for equilibrium once per step, `eqstate = model(params, structure)`, then vectorizes the constraint over that shared state. This separate evaluation path is exactly why only optimizers like `SLSQP` built to do it can enforce constraints.

## How a constraint works

A constraint lives the [same two-phase life as a goal](goals.md#goals-in-action): we construct it with a key and two bounds, and it resolves that key to a structure index on the fly at evaluation, caching nothing. It is the same stateless pytree, with bounds where a goal keeps a target and weight.

The differences are twofold.

- **Law needs an enforcer.**
Constraints are honored only by optimizers that support them, `SLSQP`, `IPOPT`, and `TrustRegionConstrained`.
Hand constraints to any other optimizer and they are politely ignored.
- **One method, different name.**
Where a goal implements `prediction`, a constraint implements `constraint`.
Same signature, same job: slice the quantity of interest for one element out of an `EquilibriumState`.

One constraint, one key, same as goals.
To bound many elements, create one constraint per element, and the machinery stacks same-type constraints into a single vectorized call.

Constraints also share the goal's `evaluate` convenience: `constraint.evaluate(datastructure)` reads the equilibrium state off a network or mesh and returns the constrained quantity, a quick way to check a bound is where we think it is before handing the constraint to an optimizer. See [trying a goal out](goals.md#trying-a-goal-out-with-evaluate) for the shape of it, keeping in mind it reads the datastructure's geometry as-is and does not solve.

## Where to next

- To write a constraint the library does not ship, head to [custom constraints](custom_constraints.md).
- To nudge a quantity toward a target instead of bounding it, see [goals](goals.md).
