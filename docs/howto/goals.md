# Goals

A goal is how we tell JAX FDM what we want a structure to become.
It is a *soft* target on a single equilibrium quantity: make this edge two meters long, pull this node onto that plane, minimize the force in this cable.
If we stack a handful of goals into a loss, then [constrained form-finding](constrained_form_finding.md) will chase them simultaneously.

This guide is about what a goal *is*: the inputs we give it, the two-phase life it leads, and how an element key becomes an array row.
Once the anatomy is clear, [custom goals](custom_goals.md) shows how to write our own, and [constraints](constraints.md) covers the "hard goal" sibling.

!!! note "New here?"

    A goal reads an attribute of interest out of an *equilibrium state*, the array bundle a form-finding solve produces.
    If the terms *model*, *structure*, and *equilibrium state* are new, skim [the numerical core](form_finding.md#the-numerical-core) and then come back.
    This guide picks up exactly where that section leaves off.

## The anatomy of a goal

Say we want an edge with `key=(6, 7)` of a network to be two meters long.
We grab `EdgeLengthGoal` and feed in the target to reach:

```python
from jax_fdm.goals import EdgeLengthGoal


edge = (6, 7)
goal = EdgeLengthGoal(edge, target=2.0, weight=1.0)
```

That single call is the entire public surface of a goal, and it is the natural place to start taking one apart.
Let us go one step at a time.

### What we pass in

Every goal is built from the same three inputs, defined once as fields on the base `Goal` and inherited by every goal in the library:

```python
import equinox as eqx


class Goal(eqx.Module):

    key: Array     # which element (resolved to an index at evaluation time)
    target: Array  # the value to drive the quantity toward
    weight: Array  # how much this goal matters in the loss
```

Those three fields are what we supply, and each has a destination:

- **`key`** names the element the goal acts on, the edge tuple `(u, v)` in the call above.
- **`target`** is the value we want the structure to reach in its equilibrium state, `2.0` here. The loss later measures how far each prediction is from it.
- **`weight`** (default `1.0`) scales this goal's contribution to the loss, so a heavier weight makes the optimizer favor satisfying this goal over lighter ones.

A goal is an [equinox](https://docs.kidger.site/equinox/) module, the same pytree flavor as the [structure](form_finding.md#the-structure) we met earlier: its fields are the leaves of a registered JAX pytree, and the constructor that fills them is synthesized from their declaration. That is why the goals we write declare no `__init__` of their own, they inherit these three fields for free, and it is what lets JAX FDM stack a list of same-type goals and vectorize them in one differentiable call (see [Goals in action](#goals-in-action)).

So instantiating a goal costs us almost nothing, which is the point.
But that begs a question: what *is* an `EdgeLengthGoal`, and what happens under the hood?

### What is under the hood

Surprisingly little. An `EdgeLengthGoal` looks like this:

```python
from jaxtyping import Array

from jax_fdm.goals.edge import EdgeGoal
from jax_fdm.goals import GoalState


class EdgeLengthGoal(EdgeGoal):
    """
    Drive an edge toward a target length.
    """

    def prediction(self, eq_state, structure, index):
        """
        The edge's current length in the equilibrium state.
        """
        return eq_state.lengths[index, 0]

    def goal(self, target, prediction):
        """
        The reference the prediction is compared against, here the target itself.
        """
        return target

    def __call__(self, eq_state, structure):
        """
        Evaluate the goal at an equilibrium state.
        """
        index = self.index(structure)
        prediction = self.prediction(eq_state, structure, index)
        goal = self.goal(self.target, prediction)

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)
```

There are four parts underpinning how any goal works:

- **The base class says *where*.** `EdgeGoal` fixes the element the goal lives on, an edge, and with it the vocabulary the goal's key is resolved against. Every goal picks one such element family, and that choice is what wires up the key resolution we would otherwise write by hand. See [Goal families](#goal-families) below for a complete taxonomy.
- **The `prediction` method says *what* and *how*.** Given an `eq_state`, the `structure`, and the `index` the edge `key` was resolved to, it returns the quantity the goal cares about, here the edge's length, read straight out of the equilibrium state's `lengths` array. That single method is what makes an `EdgeLengthGoal` an *edge length* goal rather than any other kind. Its return shape is also what fixes the goal's *rank*: one number per element makes this a scalar goal, an xyz triple would make it a vector goal. We never declare the rank separately, the prediction speaks for it.
- **The `goal` method says *toward what*.** An error term in the loss measures the gap between a goal's *prediction* and its *goal* value. Here `goal` just hands back the `target` unchanged: reach the target length, plain and simple. But it receives the current `prediction` too, and some goals use it to compare against a *moving* reference rather than a fixed target, the trick behind `NodeLineGoal` and `NodePlaneGoal`. [Custom goals](custom_goals.md#recipe-2-a-custom-vector-goal-with-a-moving-target) puts it to work.
- **The `__call__` method says *put it together*.** A goal is a callable object: `__call__` is the one method that runs the others. It resolves the element `index` from the structure, asks `prediction` for the current value, hands that to `goal` to get the reference to compare against, and bundles the two with the `weight` into a `GoalState`, a small record carrying exactly the three numbers an error term needs. We never call this ourselves, but it is the seam where a goal plugs into the rest of the library's workflow.

Here is how those pieces compose in a single evaluation. Given an `eq_state` and the `structure` it was solved on, calling the goal resolves the index, reads the quantity, resolves the reference, and packages both with the weight:

```python
goal_state = goal(eq_state, structure)   # -> GoalState(goal=..., prediction=..., weight=...)
```

A goal turns three separate concerns, *what to read*, *what to aim at*, and *how much it matters*, into one uniform `GoalState` that downstream loss code can consume without knowing anything about edges or lengths.
And downstream code is exactly a **loss**.
A loss holds a list of **errors**, which in turn hold a sequence of goals.
To score an equilibrium state, each error calls each goal via `goal(eq_state, structure)`, collects the returned `GoalState` records, and feeds their `prediction`, `goal`, and `weight` to measure the gap.
So the goal never computes an error itself. It only reports its three numbers, and the loss composes them into the single scalar the optimizer minimizes.
(We will meet the loss and its error terms in [constrained form-finding](constrained_form_finding.md).)

To recapitulate:

- The base class picks an element (edge, node, vertex, face, network, mesh).
- The fields store a `key`, `target`, and `weight`.
- The `prediction` reads and processes the quantity of interest from an equilibrium state, and its return shape sets the goal's rank.
- The `goal` method decides what to compare it against.
- `__call__` resolves the index and composes the three into a `GoalState` the error and the loss consumes.

## Goal families

The `EdgeLengthGoal` above made one choice, `EdgeGoal`, and that is the axis every goal is built along, its **element family**:

| Choice | Options |
| --- | --- |
| What element does it live on? | `NodeGoal`, `VertexGoal`, `EdgeGoal`, `FaceGoal`, `NetworkGoal`, `MeshGoal` |

The family fixes the vocabulary the goal's key is resolved against: an `EdgeGoal` resolves its key against the structure's edges, a `NodeGoal` against its nodes, and so on.

A goal's *rank*, one number per element (scalar) or one xyz vector per element (vector), is not a second choice we make up front. It falls straight out of what the `prediction` returns: hand back a lone value and the goal is scalar, hand back a triple and it is a vector. The two must agree with the target we stored, and `__call__` checks exactly that, raising a `ValueError` if a goal's prediction shape and its target shape disagree, the usual sign that a scalar target was handed to a vector goal or the other way around.

## Goals in action

A goal lives a two-phase life, and the split explains why we can build one before the FDM is executed.

**Phase one: construction.**
This is the `EdgeLengthGoal(edge, target=2.0)` call from the anatomy: we store a `key`, a `target`, and a `weight`, and nothing more.
No structure is involved yet, so we can create goals anywhere, in any order, before or after form-finding.
One goal, one key.
To target many elements, create one goal per element, and the machinery stacks same-type goals into a single vectorized call.
The exception is the [aggregate goal](custom_goals.md#recipe-3-an-aggregate-goal), which judges a group as a whole and takes the whole list.

**Phase two: evaluation.**
When we call `constrained_fdm`, each loss evaluation calls the goal against the solved `eq_state` and the [structure](form_finding.md#the-structure), the object that carries the connectivity and the index tables.
The goal resolves its element key into an integer `index` against the structure's element ordering, then reads its quantity of interest by that `index` straight out of the `EquilibriumState` the FDM produces.
We speak in keys at construction, the machinery speaks in array rows at evaluation, and the key-to-index resolution happens on the fly each time the goal is called, with no separate initialization step to remember.

!!! note "Goals are stateless"

    A goal never stores a resolved index or caches a structure. It holds only its key, target, and weight, and resolves the key afresh from whatever structure it is called against. That is what makes a goal a plain, immutable pytree: the same goal object evaluates against any compatible structure, and JAX can stack, vectorize, and differentiate through a whole list of them without any hidden state getting in the way.

## Keys versus indices

!!! tip

    **The reason for the existence of a key-or-index duality is that solvers and optimizers need consecutive integer indices for fast array access and vectorization, but datastructure keys may not respect this contract.**
    Not only that, but keys are heterogeneous: while node and vertex keys are integers, edges have pairs of integers as keys due to COMPAS datastructure semantics.
    Therefore, we have to establish a sharp separation between the use cases for a `key` (for streamlined modeling and prototyping) and an `index` (for the fast JAX numerical core) so that both entities can co-exist.
    Note that nodes, vertices, edges, and faces are indexed deterministically at runtime through their respective datastructure generators, so there **is** a transparent way to go back and forth between the two. For example, the first edge produced by running `FDMesh.edges()` will have `index=0`, while the last output edge will have `index=FDMesh.number_of_edges() - 1`.

## Where to next

- To write a goal the library does not ship, head to [custom goals](custom_goals.md).
- To impose a hard limit instead of a soft target, see [constraints](constraints.md).
