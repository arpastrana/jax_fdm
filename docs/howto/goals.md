# Goals

A goal is how you tell JAX FDM what you want a structure to become.
It is a *soft* target on a single equilibrium quantity: make this edge two meters long, pull this node onto that plane, minimize the force in this cable.
Stack a handful of goals into a loss and [constrained form-finding](constrained_form_finding.md) chases them all at once.

This guide is about what a goal *is*: the inputs you give it, the two-phase life it leads, and how an element key becomes an array row.
Once the anatomy is clear, [custom goals](custom_goals.md) shows you how to write your own, and [constraints](constraints.md) covers the hard-limit sibling.

!!! note "New here?"

    A goal reads an attribute of interest out of an *equilibrium state*, the array bundle a form-finding solve produces.
    If the terms *model*, *structure*, and *equilibrium state* are new, skim [the numerical core](form_finding.md#the-numerical-core) and then come back.
    This guide picks up exactly where that section leaves off.

## The anatomy of a goal

Say you want an edge with `key=(6, 7)` of a network to be two meters long.
You grab `EdgeLengthGoal` and feed in the target to reach:

```python
from jax_fdm.goals import EdgeLengthGoal


edge = (6, 7)
goal = EdgeLengthGoal(edge, target=2.0, weight=1.0)
```

That single call is the entire public surface of a goal, and it is the natural place to start taking one apart.
Let us go one step at a time.

### What you see

Every goal is built from the same three inputs, defined once on the base `Goal` and inherited by every goal in the library:

```python
class Goal:

    def __init__(self, key, target, weight=1.0):
        self.key = key        # which element (resolved to an index at init time)
        self.target = target  # the value to drive the quantity toward
        self.weight = weight  # how much this goal matters in the loss
```

Those three arguments are what you supply, and each has a destination:

- **`key`** names the element the goal acts on, the edge tuple `(u, v)` in the call above.
- **`target`** is the value you want the structure to reach in its equilibrium state, `2.0` here. The loss later measures how far each prediction sits from it.
- **`weight`** (default `1.0`) scales this goal's contribution to the loss, so a heavier weight makes the optimizer favor satisfying this goal over lighter ones.

So instantiating a goal costs you almost nothing, which is the point.
But that simplicity begs a question: if you never wrote a line of it, what *is* an `EdgeLengthGoal`, and what happens under the hood?

### What is under the hood

Surprisingly little. Spelled out in full, an `EdgeLengthGoal` looks like this:

```python
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal
from jax_fdm.goals import GoalState


class EdgeLengthGoal(ScalarGoal, EdgeGoal):
    """
    Drive an edge toward a target length.
    """

    def __init__(self, key, target, weight=1.0):
        self.key = key        # which edge (resolved to an index at init time)
        self.target = target  # the length to drive the edge toward
        self.weight = weight  # how much this goal matters in the loss

    def prediction(self, eq_state, index):
        """
        The edge's current length in the equilibrium state.
        """
        return eq_state.lengths[index]

    def goal(self, target, prediction):
        """
        The reference the prediction is compared against, here the target itself.
        """
        return target

    def __call__(self, eq_state):
        """
        Evaluate the goal at an equilibrium state.
        """
        prediction = self.prediction(eq_state, self.index)
        goal = self.goal(self.target, prediction)

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)
```

Five moving parts underpin how any goal works:

- **The base classes say *what* and *where*.** `ScalarGoal` fixes the shape of the quantity, one number per element, and `EdgeGoal` fixes the element it lives on, an edge. Every goal picks one from each of these two families, and that pair is what wires up the target storage and the key resolution you would otherwise write by hand.
- **The constructor says *which* and *how much*.** It stores the three inputs from the section above: the `key` of the edge, the `target` length, and the `weight`. Most goals in the library skip writing this out and simply inherit it from the base `Goal`, since it is the same three lines every time. We spell it out here to show where your inputs land.
- **The `prediction` method says *how*.** Given an `eq_state` and the `index` the edge `key` was resolved to, it returns the quantity the goal cares about, here the edge's length, read straight out of the equilibrium state's `lengths` array. That single method is what makes an `EdgeLengthGoal` an *edge length* goal rather than any other kind.
- **The `goal` method says *toward what*.** An error term in the loss measures the gap between a goal's *prediction* and its *goal* value. Here `goal` just hands back the `target` unchanged: reach the target length, plain and simple. But it receives the current `prediction` too, and some goals use it to compare against a *moving* reference rather than a fixed target, the trick behind `NodeLineGoal` and `NodePlaneGoal`. [Custom goals](custom_goals.md#recipe-2-a-custom-vector-goal-with-a-moving-target) puts it to work.
- **The `__call__` method says *put it together*.** A goal is a callable object: `__call__` is the one method that runs the other two. It asks `prediction` for the current value, hands that to `goal` to get the reference to compare against, and bundles the two with the `weight` into a `GoalState`, a small record carrying exactly the three numbers an error term needs. You never call this yourself, but it is the seam where a goal plugs into the rest of the machinery.

Here is how those pieces compose in a single evaluation. Given an `eq_state`, calling the goal reads the quantity, resolves the reference, and packages both with the weight:

```python
goal_state = goal(eq_state)   # -> GoalState(goal=..., prediction=..., weight=...)
```

That is the whole point of making a goal callable: it turns three separate concerns, *what to read*, *what to aim at*, and *how much it matters*, into one uniform `GoalState` that downstream code can consume without knowing anything about edges or lengths.
And downstream code is exactly a **loss**.
A loss holds a list of goals, and to score an equilibrium state it calls each goal in turn, `goal(eq_state)`, collects the returned `GoalState` records, and feeds their `prediction`, `goal`, and `weight` into an error term that measures the gap.
So the goal never computes an error itself. It only reports its three numbers, and the loss composes them into the single scalar the optimizer minimizes.
(You will meet the loss and its error terms in [constrained form-finding](constrained_form_finding.md).)

That is the whole contract, top to bottom:

- The base classes pick a shape and an element.
- The constructor stores a `key`, `target`, and `weight`.
- The `prediction` reads and processes the quantity of interest from an equilibrium state.
- The `goal` methods decides what to compare it against.
- `__call__` composes the three into a `GoalState` the loss consumes.

## Goal families

The `EdgeLengthGoal` above made two choices, `ScalarGoal` and `EdgeGoal`, and those are the two axes every goal is built along:

| Choice | Options |
| --- | --- |
| What shape is the quantity? | `ScalarGoal` (one number per element) or `VectorGoal` (one xyz vector per element) |
| What element does it live on? | `NodeGoal`, `VertexGoal`, `EdgeGoal`, `FaceGoal`, `NetworkGoal`, `MeshGoal` |

The shape choice also polices your target: a scalar goal stores whatever you pass as one number per element, while a vector goal expects an xyz triple and stops you with a `TypeError` if you hand it a lone number, a nudge toward the scalar variant you probably meant.

## Goals in action

The anatomy showed a goal's pieces.
Now it is time to see how they come to life.
A goal lives a two-phase life, and the split explains why you can build one long before a solve exists.

**Phase one: construction.**
This is the `EdgeLengthGoal(edge, target=2.0)` call from the anatomy: you store a `key`, a `target`, and a `weight`, and nothing more.
No structure is involved yet, so you can create goals anywhere, in any order, before or after form-finding.
One goal, one key: to target many elements, create one goal per element, and the machinery batches same-type goals into a single vectorized call.
(The exception is the [aggregate goal](custom_goals.md#recipe-3-an-aggregate-goal), which judges a group as a whole and takes the whole list.)

**Phase two: initialization.**
When you call `constrained_fdm`, the optimizer calls `goal.init(model, structure)` behind the scenes.
This resolves your element key into an integer `index` inside a [structure](form_finding.md#the-structure)'s topology, the object that carries the connectivity and the index tables.
Once the `index` is known, the goal reads its quantity of interest by `index` straight out of the `EquilibriumState` the FDM produces.
You speak in keys at construction, the machinery speaks in array rows at evaluation, and `init` is the translation step between the two.

## Keys versus indices

!!! tip

    **The reason for the existence of a key-or-index duality is that solvers and optimizers need consecutive integer indices for fast array access and vectorization, but datastructure keys may not respect this contract.**
    Not only that, but keys are heterogeneous: while node and vertex keys are integers, edges have pairs of integers as keys due to COMPAS datastructure semantics.
    Therefore, we have to establish a sharp separation between the use cases for a `key` (for streamlined modeling and prototyping) and an `index` (for the fast JAX numerical core) so that both entities can co-exist.
    Note that nodes, vertices, edges, and faces are indexed deterministically at runtime through their respective datastructure generators, so there **is** a transparent way to go back and forth between the two. For example, the first edge produced by running `FDMesh.edges()` will have `index=0`, while the last output edge will have `index=FDMesh.number_of_edges() - 1`.

## Where to next

- To write a goal the library does not ship, head to [custom goals](custom_goals.md).
- To impose a hard limit instead of a soft target, see [constraints](constraints.md).
