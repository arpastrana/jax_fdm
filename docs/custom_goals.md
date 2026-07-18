# Custom goals and constraints

JAX FDM ships with a rich bank of goals and constraints, but sooner or later your design brief will ask for something the bank does not have.
Maybe you want an edge to rise by exactly half a meter.
Maybe you want a node to sit on a sphere.
Maybe your PhD committee wants both, by Friday.

Good news: a custom goal is one class and one method away.
This guide walks you through five recipes, from a two-line scalar goal to a full custom constraint.

## How a goal works

A goal lives a two-phase life.

**Phase one is construction.**
You build the goal with an element key, a target, and a weight.
One goal, one key: to target many elements, create one goal per element, and the machinery batches same-type goals into a single vectorized call.
(The exception is the aggregate goal of recipe 3, which judges a group as a whole and takes the whole list.)
No structure is involved yet, so you can create goals anywhere, in any order, before or after form-finding.

**Phase two is initialization.**
When you call `constrained_fdm`, the optimizer calls `goal.init(model, structure)` behind the scenes.
This resolves your element key (a node key, an edge tuple, a vertex key) into an integer index inside the equilibrium structure.
From then on, the goal reads its quantity of interest straight out of an `EquilibriumState` by index.
In other words: you speak in keys at construction, the machinery speaks in array rows at evaluation, and `init` is the translation step between the two.

Every goal you write is the product of two choices:

| Choice | Options |
| --- | --- |
| What shape is the quantity? | `ScalarGoal` (one number per element) or `VectorGoal` (one xyz vector per element) |
| What element does it live on? | `NodeGoal`, `VertexGoal`, `EdgeGoal`, `FaceGoal`, `NetworkGoal`, `MeshGoal` |

Mix one from each row, add a `prediction` method, and you have a goal.
The base classes handle everything else: target storage, weighting, vectorization over many elements, and the squared error against your loss function.

## Recipe 1: a custom scalar goal

Suppose you care about how much an edge climbs, its vertical rise.
No goal in the bank measures that, so we write one.

```python
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeRiseGoal(ScalarGoal, EdgeGoal):
    """
    Drive the vertical rise of an edge toward a target value.
    """

    def prediction(self, eq_state, index):
        vector = eq_state.vectors[index, :]

        return jnp.abs(vector[2])
```

That is the whole class.
Let's unpack the two lines that matter.

- `eq_state` is an `EquilibriumState`, a named tuple holding the solved geometry: `xyz` (node coordinates), `residuals`, `lengths`, `forces`, `loads`, and `vectors` (edge vectors from tail to head).
Your prediction slices what it needs out of these arrays.
- `index` is where phase two landed your key: the integer row of *one* edge inside those arrays.
You write the prediction for a single element, and JAX FDM vectorizes it across every edge you attach the goal to.
Free performance, no loops.

Now use it like any built-in goal:

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import EdgeForceDensityParameter


goals = [EdgeRiseGoal(edge, target=0.5) for edge in network.edges()]
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
    No Python `if` statements on array values, no printing, no mutating outside state.
    In exchange, your goal is differentiated, JIT-compiled, and vectorized for free.

## Recipe 2: a custom vector goal with a moving target

Scalar goals compare one number against one number.
Vector goals compare xyz against xyz.
And sometimes the target is not a fixed point but a *set*, like the surface of a sphere, where the reference point depends on where the node currently is.

That is what the `goal` method is for.
It receives the stored target and the current prediction, and returns the reference value the prediction is compared against.
The built-in `NodeLineGoal` and `NodePlaneGoal` use this trick to chase closest points.
Here we pull a node onto a sphere:

```python
import jax.numpy as jnp

from jax_fdm.geometry import normalize_vector
from jax_fdm.goals import VectorGoal
from jax_fdm.goals.node import NodeGoal


class NodeSphereGoal(VectorGoal, NodeGoal):
    """
    Pull a node onto the surface of a sphere.

    The target packs the sphere as four numbers: the center xyz and the radius.
    """

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = jnp.reshape(jnp.asarray(target), (-1, 4))

    def prediction(self, eq_state, index):
        return eq_state.xyz[index, :]

    def goal(self, target, prediction):
        center = target[:3]
        radius = target[3]
        direction = normalize_vector(prediction - center)

        return center + radius * direction
```

```python
center = [5.0, 0.0, 0.0]
radius = 5.0

goals = [NodeSphereGoal(node, target=[*center, radius]) for node in free_nodes]
```

Two things to notice.

First, the sphere travels inside `target` as a flat pack of four numbers, and the `target` property reshapes it to one row per element.
Overriding the target property is how a goal declares a custom target shape, exactly like `NodeLineGoal` reshapes its two line points to `(-1, 2, 3)`.

Second, you might be tempted to store the center and radius as plain attributes instead, say `self.radius`, and skip the property.
Resist.
Behind the scenes, same-type goals are batched into one vectorized call by rebuilding the goal *from its init signature*: every `__init__` parameter must be stored as a same-named attribute so the machinery can read it back and stack it.
Break that and you get an `AttributeError` quoting the rule at you on the first optimization run.
Packing the sphere into `target` sidesteps the question entirely, because `target` already rides the existing machinery.

## Recipe 3: an aggregate goal

The goals so far judge each element on its own.
An aggregate goal judges a *group* as a whole: the total length of a cable bundle, the evenness of a strip of nodes, the variance of a family of forces.

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
goal = EdgesTotalLengthGoal(key=list(network.edges()), target=12.0)
```

`is_aggregate = True` does two things for you.
It unlocks the list of keys (a per-element goal insists on exactly one key, and tells you so with a `TypeError`), and it tells the vectorizer to hand your prediction the *whole* row of indices in a single call, instead of one index at a time.
`index` is then an array of edge positions, and you reduce across it however your quantity demands: a sum here, a variance in `EdgesLengthEqualGoal`, a colinearity energy in `NodesColinearGoal`.
The whole-structure goals (`Network*`, `Mesh*`) are aggregates too, just ones whose group is the entire structure, so they take no key list at all.

## Recipe 4: retargeting a goal from nodes to vertices

JAX FDM speaks two dialects.
Networks have **nodes**, meshes have **vertices**, and goals are picky about which one they hear.
Aim a `Node*` goal at a mesh and you get a `TypeError` telling you to use the `Vertex*` counterpart (and the other way around).
This is deliberate: the two vocabularies resolve keys through different index tables, and silently mixing them is the stuff of subtle bugs.

So what if you wrote `NodeSphereGoal` and now want it on a mesh?
You write its vertex twin, and the twin is gloriously boring:

```python
from jax_fdm.goals.vertex import VertexGoal


class VertexSphereGoal(VertexGoal, NodeSphereGoal):
    """
    Pull a mesh vertex onto the surface of a sphere.
    """
```

No methods.
No body.
Just a docstring and an inheritance list, with `VertexGoal` first so the vertex key resolution wins the method resolution order.
The prediction, the target property, and the sphere projection all come along for the ride.

Every vertex goal in the library (`VertexPointGoal`, `VertexLineGoal`, `VertexResidualForceGoal`, and friends) is built exactly this way, so you are in good company.

## Recipe 5: a custom constraint

Goals express preference: get as close to the target as you can.
Constraints express law: stay between these bounds, or the optimizer does not go home.
They are enforced by optimizers that support them, like `SLSQP` and `IPOPT`.

A custom constraint subclasses an element family and implements one method, `constraint`, which plays the same role as a goal's `prediction`:

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

The bounds arrive through the constructor as `bound_low` and `bound_up`.
Leave one as `None` and it becomes an infinity of the appropriate sign, so one-sided constraints cost you nothing.

## One last rule

Speak the right vocabulary: `Node*` classes on networks, `Vertex*` classes on meshes, no exceptions.
When you need a goal on the other side of the fence, write a thin twin as in recipe 4.

And that is it.
One class, one method, and the entire differentiable machinery (gradients, JIT compilation, vectorization) adopts your goal as one of its own.
If you build a goal or constraint that others might want, contributions to the bank are warmly welcome 🚀.
