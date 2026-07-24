# Custom goals

JAX FDM ships with a rich bank of goals, but sooner or later a structural design problem will ask for something the bank does not have.
Maybe we want an edge to carry a target force, whether it pulls or pushes.
Maybe we want a node to hover directly above a point on the ground.

This guide walks us through five recipes for custom goal-making, from a two-line scalar goal to a goal that chases a moving target and one that carries reference data of its own.
It assumes you have read [goals](goals.md), which lays out the anatomy the recipes below build on.

## Recipe 1: A custom scalar goal

Suppose we care about the *magnitude* of the force in an edge, tension or compression alike, and want to drive it toward a target value.
The built-in `EdgeForceGoal` targets the signed force, so a positive target means tension and a negative one compression.
A magnitude goal that ignores the sign is not in the bank, so we write one from scratch.

```python
import jax.numpy as jnp

from jax_fdm.goals.edge import EdgeGoal


class EdgeForceMagnitudeGoal(EdgeGoal):
    """
    Drive the absolute internal force of an edge toward a target value.
    """

    def prediction(self, eq_state, structure, index):
        return jnp.abs(eq_state.forces[index, 0])
```

That is the whole class.
We subclass a single element family, `EdgeGoal`, which fixes the goal to edges and resolves its key against them. There is no rank to declare: the prediction returns one number per edge, so the goal is scalar automatically.
Let us unpack the three arguments the prediction receives.

- `eq_state` is an [`EquilibriumState`](form_finding.md#the-equilibrium-state), the array bundle a form-finding solve produces. It holds the solved geometry and force state: `xyz`, `residuals`, `lengths`, `forces`, `loads`, and `vectors`. Our prediction slices what it needs out of these arrays, here the edge `forces`.
- `structure` is the [structure](form_finding.md#the-structure) the goal is evaluated against, carrying the connectivity and topology. Most goals ignore it, ours included, but it is there when a goal needs to reach for whole-structure data.
- `index` is the integer row our edge key resolves to against the structure: the row of *one* edge inside those arrays.
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

from jax_fdm.goals.node import NodeGoal


class NodeVerticalLineGoal(NodeGoal):
    """
    Pull a node onto the vertical line through a target point.
    """

    def prediction(self, eq_state, structure, index):
        return eq_state.xyz[index, :]

    def goal(self, target, prediction):
        target_xy = target[:2]        # the line's x and y
        node_z = prediction[2:]       # the node's own height

        return jnp.concatenate((target_xy, node_z))
```

This one is a *vector* goal, and again we never say so out loud: the prediction returns the node's three coordinates, so both prediction and target are xyz triples and the goal is a vector goal by construction.

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

from jax_fdm.goals.edge import EdgeGoal


class EdgesTotalLengthGoal(EdgeGoal):
    """
    Drive the combined length of a group of edges toward a target value.
    """

    is_aggregate = True

    def prediction(self, eq_state, structure, index):
        lengths = eq_state.lengths[index]

        return jnp.sum(lengths)
```

```python
goal = EdgesTotalLengthGoal(list(network.edges()), target=12.0)
```

`is_aggregate = True` does two things for us.
It marks the goal as taking a whole list of keys rather than a single one (hand a per-element goal a list and it stores it, then trips a shape error at evaluation once the mismatch is known), and it tells the machinery to hand our prediction the *whole* row of indices in a single call, instead of one index at a time.
`index` is then an array of edge positions, and we reduce across it however our quantity demands: a sum here, a variance in `EdgesLengthEqualGoal`, a colinearity energy in `NodesColinearGoal`.
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

Much of the vertex goal family in the library (`VertexPointGoal`, `VertexLineGoal`, `VertexResidualForceGoal`, and friends) is built exactly this way, an empty twin of a `Node*` goal. The exceptions are the vertex goals with genuinely mesh-only behavior, such as `VertexNormalAngleGoal`, which have no network counterpart to inherit from and so write their own `prediction`.

## Recipe 5: A goal that carries its own data

Every goal so far got by on the three fields the base gives us: `key`, `target`, `weight`.
But some goals need to *carry an extra piece of data* of their own.
Consider driving the angle between an edge and a fixed reference direction toward a target: the goal has to remember which direction to measure against, and that direction is neither the target nor the key.
It is a fourth field, stored on the goal alongside the other three.

Because a goal is an [equinox](https://docs.kidger.site/equinox/) module, adding a field is a one-liner, but there is a subtlety in *how* we declare it. Here is the library's own `EdgeAngleGoal`:

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import angle_vectors
from jax_fdm.goals.edge import EdgeGoal


class EdgeAngleGoal(EdgeGoal):
    """
    Drive the angle between an edge and a reference vector toward a target.
    """

    vector: Float[Array, "3"] = eqx.field(kw_only=True)

    def __init__(self, key, target, weight=1.0, *, vector):
        self.key = key
        self.target = target
        self.weight = weight
        self.vector = jnp.asarray(vector)

    def prediction(self, eq_state, structure, index):
        return angle_vectors(eq_state.vectors[index, :], self.vector)
```

Two things earn an explanation.

- **The new field is `kw_only=True`.** Our `vector` is *required*, it has no sensible default, but the base already gives `weight` a default of `1.0`. Plain field order would then put a required field after a defaulted one, which Python's dataclass rules forbid ("non-default argument follows default argument"). Marking `vector` keyword-only sidesteps the ordering rule entirely: it lives off in its own keyword-only slot, so it can stay required while `weight` keeps its default. That is why we construct the goal as `EdgeAngleGoal(edge, target=0.0, vector=[0, 0, 1])`, with `vector` always named.
- **We write an explicit `__init__`.** Equinox would happily synthesize one, so why not lean on it as the other recipes do? Because `vector` is a differentiable array leaf, and hiding it from the constructor (with `init=False`) would make equinox warn that a float leaf is left unset. Writing the constructor ourselves lets us both accept `vector` as an argument and coerce it (here `jnp.asarray`) on the way in.

The `prediction` then reads its stored `vector` straight off `self`, exactly as it reads its `index` from the structure.
This is the pattern for any goal that needs reference data of its own: a plane's normal, a set of anchor points, a per-element weight map.
Declare it as a `kw_only=True` field, set it in a small `__init__`, and read it off `self` in the prediction.

!!! tip "When a simple field is enough"

    If the extra data has a natural default, a plain `eqx.field(default=...)` needs neither `kw_only` nor a hand-written `__init__`, equinox synthesizes the constructor and the field slots in after `weight` without complaint.
    The `kw_only` dance is specifically for data that is *required* and has *no default*, which is the common case for a reference the goal cannot invent on its own.

## Summary

One class, one method, and the entire differentiable machinery (gradients, JIT compilation, vectorization) adopts our goal as one of its own.
When a preference is not enough and you need to draw a hard line instead, head over to [custom constraints](custom_constraints.md).
And if you build a goal that others might want, remember that contributions are warmly welcome 🚀.
