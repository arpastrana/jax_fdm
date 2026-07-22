# Custom goals

JAX FDM ships with a rich bank of goals, but sooner or later a structural design problem will ask for something the bank does not have.
Maybe we want an edge to carry a target force, whether it pulls or pushes.
Maybe we want a node to hover directly above a point on the ground.

This guide walks us through four recipes for custom goal-making, from a two-line scalar goal to a goal that chases a moving target.
It assumes you have read [goals](goals.md), which lays out the anatomy the recipes below build on.

## Recipe 1: A custom scalar goal

Suppose we care about the *magnitude* of the force in an edge, tension or compression alike, and want to drive it toward a target value.
The built-in `EdgeForceGoal` targets the signed force, so a positive target means tension and a negative one compression.
A magnitude goal that ignores the sign is not in the bank, so we write one from scratch.

```python
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeForceMagnitudeGoal(ScalarGoal, EdgeGoal):
    """
    Drive the absolute internal force of an edge toward a target value.
    """

    def prediction(self, eq_state, index):
        return jnp.abs(eq_state.forces[index])
```

That is the whole class.
Let us unpack the two arguments that matter.

- `eq_state` is an [`EquilibriumState`](form_finding.md#the-equilibrium-state), the array bundle a form-finding solve produces. It holds the solved geometry and force state: `xyz`, `residuals`, `lengths`, `forces`, `loads`, and `vectors`. Our prediction slices what it needs out of these arrays, here the edge `forces`.
- `index` is the integer row our edge key was resolved to when the goal was initialized: the row of *one* edge inside those arrays.
We write the prediction for a single element, and JAX FDM vectorizes it across every edge we attach the goal to.

Now we use it like any built-in goal:

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter


goals = [EdgeForceMagnitudeGoal(edge, target=0.5) for edge in network.edges()]
loss = Loss(SquaredError(goals=goals))
parameters = [EdgeForceDensityParameter(edge, -20.0, -0.1) for edge in network.edges()]

network = constrained_fdm(
    network,
    optimizer=LBFGSB(),
    loss=loss,
    parameters=parameters,
    maxiter=200,
)
```

!!! tip "Keep predictions pure"

    A prediction runs inside JAX's tracing machinery, so it must be a pure function of its inputs.
    Use `jnp` operations only.
    No Python `if` statements on array values, no `for` loops, no printing, no mutating outside state.
    In exchange, our goal is differentiated, JIT-compiled, and vectorized for free.

## Recipe 2: A custom vector goal with a moving target

Scalar goals compare one number against one number.
Vector goals compare a 3D vector against another 3D vector.
And sometimes the target is not a fixed point but a *set*, like a line, where the reference point depends on where the node currently is.

That is what the `goal` method is for.
It receives the stored target and the current prediction, and returns the reference value the prediction is compared against.
The built-in `NodeLineGoal` and `NodePlaneGoal` use this trick to chase closest points.
Here we pull a node onto a vertical line through a target point, so the node ends up directly above (or below) the point without caring at what height:

```python
import jax.numpy as jnp

from jax_fdm.goals import VectorGoal
from jax_fdm.goals.node import NodeGoal


class NodeVerticalLineGoal(VectorGoal, NodeGoal):
    """
    Pull a node onto the vertical line through a target point.
    """

    def prediction(self, eq_state, index):
        return eq_state.xyz[index, :]

    def goal(self, target, prediction):
        target_xy = target[:2]        # the line's x and y
        node_z = prediction[2:]       # the node's own height

        return jnp.concatenate((target_xy, node_z))
```

```python
goals = [NodeVerticalLineGoal(node, target=[2.0, 3.0, 0.0]) for node in network.nodes_free()]
```

The `goal` method builds the reference point by splicing the target's xy with the node's *own* z, so the reference is `(target_x, target_y, node_z)`.
The error the loss then measures, `prediction - goal`, is `(node_x - target_x, node_y - target_y, 0)`: it pulls the node's horizontal position toward the target and leaves its height untouched.
The target's z coordinate is ignored, so any value does.
Because the reference tracks the node's current height as it moves, the target behaves like a vertical line rather than a single point.

## Recipe 3: An aggregate goal

The goals we have seen so far assess one entity at a time.
An aggregate goal judges a *group* as a whole, such as the total length of a cable represented by a chain of edges, the evenness of a strip of nodes, and the variance of a group of edge forces.

Aggregates deviate from the standard recipe by one line:

```python
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgesTotalLengthGoal(ScalarGoal, EdgeGoal):
    """
    Drive the combined length of a group of edges toward a target value.
    """

    is_aggregate = True

    def prediction(self, eq_state, index):
        lengths = eq_state.lengths[index]

        return jnp.sum(lengths)
```

```python
goal = EdgesTotalLengthGoal(list(network.edges()), target=12.0)
```

`is_aggregate = True` does two things for us.
It unlocks the list of keys (a per-element goal insists on exactly one key, and tells us so with a `TypeError`), and it tells the vectorizer to hand our prediction the *whole* row of indices in a single call, instead of one index at a time.
`index` is then an array of edge positions (`numpy` broadcasting rules apply), and we reduce across it however our quantity demands: a sum here, a variance in `EdgesLengthEqualGoal`, a colinearity energy in `NodesColinearGoal`.
The whole-structure goals (`Network*`, `Mesh*`) are aggregates too, just ones whose group is the entire structure, so they take no key list at all.

## Recipe 4: Retargeting a goal from nodes to vertices

JAX FDM speaks two dialects for its datastructures.
Networks have **nodes**, meshes have **vertices**, and goals are picky about which one they hear.
Aim a `Node*` goal at a mesh and we get a `TypeError` telling us to use the `Vertex*` counterpart (and the other way around).
This is deliberate: the two vocabularies resolve keys through different index tables, and silently mixing them is the stuff of subtle bugs.

So what if we wrote `NodeVerticalLineGoal` and now want it on a mesh?
We write its vertex twin, and the twin is gloriously boring:

```python
from jax_fdm.goals.vertex import VertexGoal


class VertexVerticalLineGoal(VertexGoal, NodeVerticalLineGoal):
    """
    Pull a mesh vertex onto the vertical line through a target point.
    """
```

No methods and no body.
Just a docstring and an inheritance list, with `VertexGoal` first so the vertex key resolution wins the method resolution order.
The prediction and the vertical-line projection come along for the ride.

Every vertex goal in the library (`VertexPointGoal`, `VertexLineGoal`, `VertexResidualForceGoal`, and friends) is built exactly this way.

## Summary

One class, one method, and the entire differentiable machinery (gradients, JIT compilation, vectorization) adopts our goal as one of its own.
When a preference is not enough and you need to draw a hard line instead, head over to [custom constraints](custom_constraints.md).
And if you build a goal that others might want, remember that contributions are warmly welcome 🚀.
