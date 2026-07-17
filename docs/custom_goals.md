# Custom goals and constraints

JAX FDM ships with a rich bank of goals and constraints, but sooner or later your design brief will ask for something the bank does not have.
Maybe you want an edge to rise by exactly half a meter.
Maybe you want a node to sit on a sphere.
Maybe your PhD committee wants both, by Friday.

Good news: a custom goal is one class and one method away.
This guide walks you through five recipes, from a two-line scalar goal to a full custom constraint, and closes with the three contracts that keep the optimizer happy.

## How a goal works

A goal lives a two-phase life.

**Phase one is construction.**
You build the goal with an element key, a target, and a weight.
No structure is involved yet, so you can create goals anywhere, in any order, before or after form-finding.

**Phase two is initialization.**
When you call `constrained_fdm`, the optimizer calls `goal.init(model, structure)` behind the scenes.
This resolves your element key (a node key, an edge tuple, a vertex key) into an integer index inside the equilibrium structure.
From then on, the goal reads its quantity of interest straight out of an `EquilibriumState` by index.

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
        rise = jnp.abs(vector[2])

        return jnp.atleast_1d(rise)
```

That is the whole class.
Let's unpack the three lines that matter.

- `eq_state` is an `EquilibriumState`, a named tuple holding the solved geometry: `xyz` (node coordinates), `residuals`, `lengths`, `forces`, `loads`, and `vectors` (edge vectors from tail to head).
Your prediction slices what it needs out of these arrays.
- `index` is the integer position of *one* edge.
You write the prediction for a single element, and JAX FDM vectorizes it across every edge you attach the goal to.
Free performance, no loops.
- `jnp.atleast_1d` is not decoration.
A scalar prediction must have shape `(1,)`, not `()`.
Forget it and the goal raises a `ValueError` that tells you exactly this (we have all been there, so the error message is friendly).

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
Resist for one more section.
The reason is the collection contract, and it is contract number two at the end of this guide.

## Recipe 3: an aggregate goal

The goals so far judge each element on its own.
An aggregate goal judges a *group* as a whole: the total length of a cable bundle, the evenness of a strip of nodes, the variance of a family of forces.

Aggregates need three small deviations from the standard recipe:

```python
import jax.numpy as jnp
import numpy as np

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgesTotalLengthGoal(ScalarGoal, EdgeGoal):
    """
    Drive the combined length of a group of edges toward a target value.
    """

    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)
        self.is_collectible = False

    def init(self, model, structure):
        self.index = np.atleast_2d(super().index_from_structure(structure))

    def prediction(self, eq_state, index):
        lengths = eq_state.lengths[index]

        return jnp.atleast_1d(jnp.sum(lengths))
```

```python
goal = EdgesTotalLengthGoal(key=list(network.edges()), target=12.0)
```

The three deviations, in order of appearance:

1. **`self.is_collectible = False`.**
Ordinary goals of the same type get stacked into one vectorized batch.
An aggregate is already a batch of its own, so it opts out.
2. **A custom `init` that wraps the index in `np.atleast_2d`.**
The extra dimension tells the vectorizer to hand your prediction the *whole* row of indices in a single call, instead of one index at a time.
3. **A prediction over the group.**
`index` is now an array of edge positions, and you reduce across it however your quantity demands: a sum here, a variance in `EdgesLengthEqualGoal`, a colinearity energy in `NodesColinearGoal`.

!!! warning "Call `self.index_from_structure`, or `super()`, but know why"

    In this recipe `super().index_from_structure(structure)` is fine because `EdgesTotalLengthGoal` will never be subclassed for another element family.
    If your aggregate might grow a vertex twin one day (like `NodesColinearGoal` did), call `self.index_from_structure(structure)` instead, so the subclass's own resolution wins.
    Method resolution order giveth, and method resolution order taketh away.

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
Note that a constraint's value may return a plain scalar of shape `()`: constraints are flattened after vectorization, so the `(1,)` rule from recipe 1 is a goals-only affair.

## The three contracts

Three rules keep your custom classes on speaking terms with the optimizer.
Break one and you get a loud, descriptive error rather than a wrong answer, but why not skip the detour.

**Contract 1: scalar predictions have shape `(1,)`.**
A scalar goal's prediction must return shape `(1,)`, never a bare `()`.
Close every scalar prediction with `jnp.atleast_1d` and move on with your life.
The error, should you forget, names this exact fix.

**Contract 2: every init parameter is a same-named attribute.**
When many goals of one type are batched into a collection, the collection rebuilds the goal *from its init signature*: for each parameter name in `__init__`, it reads `self.<that name>` off every goal and stacks the values.
So if your `__init__` takes `stiffness`, then `self.stiffness` must exist, spelled exactly like the parameter.
Stash it as `self._stiffness` or `self.k` and the collection machinery raises an `AttributeError` quoting this rule back at you.
This is also why recipe 2 packed the sphere into `target` instead of loose attributes: `target` is already part of the init contract, so the pack rides the existing machinery with zero extra ceremony.

**Contract 3: speak the right vocabulary.**
`Node*` classes on networks, `Vertex*` classes on meshes, no exceptions.
When you need a goal on the other side of the fence, write a thin twin as in recipe 4.

And that is it.
One class, one method, and the entire differentiable machinery (gradients, JIT compilation, vectorization) adopts your goal as one of its own.
If you build a goal or constraint that others might want, contributions to the bank are warmly welcome 🚀.
