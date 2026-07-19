# Constraints

Goals express *soft* preference: get as close to the target as you can.
Constraints express *hard* restrictions: stay between these bounds, or the optimizer does not go home.

This guide is about what a constraint *is* and how it differs from a goal.
It assumes you have read [goals](goals.md); constraints reuse most of that machinery, so here we focus on what changes.
Once the anatomy is clear, [custom constraints](custom_constraints.md) shows you how to write your own, and for how goals and constraints slot into a full optimization problem, see [constrained form-finding](constrained_form_finding.md).

## The anatomy of a constraint

We take a constraint apart the same top-down way the [goals](goals.md#the-anatomy-of-a-goal) guide takes apart `EdgeLengthGoal`.
Suppose that instead of *nudging* a quantity toward a target, you want a hard floor and ceiling: no node in the network may sink below zero or rise above three.
You grab `NodeZCoordinateConstraint`, one per node, and hand each a pair of bounds:

```python
from jax_fdm.constraints import NodeZCoordinateConstraint


constraints = [NodeZCoordinateConstraint(node, 0.0, 3.0) for node in network.nodes()]
```

One constraint per node, each pinning that node's height between 0.0 and 3.0.
Set that call beside a goal's and the first difference is already visible: where the goal took a `target`, the constraint takes two bounds.

### What you see

Every constraint is built from a key and two bounds, defined once on the base `Constraint` and inherited by every constraint in the library:

```python
class Constraint:

    def __init__(self, key, bound_low=None, bound_up=None):
        self.key = key              # which element (resolved to an index at init time)
        self.bound_low = bound_low  # lower limit; None becomes -inf
        self.bound_up = bound_up    # upper limit; None becomes +inf
```

The `key` behaves exactly as a goal's: stored as given, the node key here, then resolved to an integer `index` when `constrained_fdm` calls `init`.
But **the target and weight are gone**, replaced by two bounds:

- **`bound_low`** and **`bound_up`** are the limits the optimizer must keep the quantity between. Pass `None` for either and it normalizes to the matching infinity, so a one-sided constraint (a floor with no ceiling, say) costs you nothing.
- There is **no weight**, because a constraint is not weighed against anything. It is satisfied or the solution is rejected, full stop.

### What is under the hood

Even less than a goal. Here is the whole class, straight from the library:

```python
from jax_fdm.constraints.node import NodeConstraint


class NodeZCoordinateConstraint(NodeConstraint):
    """
    Bound the Z coordinate of a node between a lower and an upper value.
    """

    def constraint(self, eq_state, index):
        return eq_state.xyz[index, 2]
```

Two moving parts, mirroring the goal with two deliberate differences:

- **One base class, not two.** A constraint subclasses only an element family (`NodeConstraint`). There is no scalar-versus-vector choice to make, because a constraint has no target shape to declare: its quantity is simply whatever number `constraint` returns.
- **The `constraint` method is the one thing you write.** It plays the exact role `prediction` plays for a goal, same signature, same job: given an `eq_state` and the `index` your key resolved to, return the quantity of interest, here the node's Z coordinate pulled from the `xyz` array. Only the name differs.

So a constraint is a goal with the target-and-weight swapped for bounds and a single-family base class.
The next section makes those differences precise.

## How a constraint works

A constraint lives the same two-phase life as a goal.
You construct it with an element key and two bounds, and when you call `constrained_fdm`, its `init` resolves the key to an integer index inside the equilibrium structure.
Keys at construction, array rows at evaluation, `init` in between — the [same translation step](goals.md#goals-in-action) goals go through.

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

## Where to next

- To write a constraint the library does not ship, head to [custom constraints](custom_constraints.md).
- To nudge a quantity toward a target instead of bounding it, see [goals](goals.md).
