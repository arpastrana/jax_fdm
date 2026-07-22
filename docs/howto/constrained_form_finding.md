# Constrained form-finding

Plain [form-finding](form_finding.md) provides us with *one* funicular geometry for given input parameters.
But a design brief rarely asks for just any equilibrium state.
Most often that not, it asks for a gridsheel whose member lengths stay between three-quarters of a meter and a meter, or a roof that touches down exactly on four given points, but where the magnitude of the reaction force at one of the supports must reach a target value.

Constrained form-finding is how we guide equilibrium computation towards *that* shape.
We describe a design specification we want as a loss to minimize and a set of bounds to respect, and `constrained_fdm` searches the space of force densities (and, if we like, supports and loads) for the geometry and the equilibrium configuration that satisfies them.

```python
from jax_fdm.equilibrium import constrained_fdm


optimized = constrained_fdm(datastructure, optimizer, loss, parameters, constraints)
```

Under the hood `constrained_fdm` runs the same force density solve as `fdm`, repeatedly inside an optimization loop.
Each iteration nudges the parameters, re-solves for equilibrium, and scores the result against our loss.
Because the whole solve is differentiable, the optimizer gets exact gradients for free to adjust the FDM parameters at each iteration until convergence.

The rest of this guide is a tour of the five ingredients we assemble, from the parameters the optimizer moves to the algorithm that moves them.

## The five ingredients

### 1. Parameters: what the optimizer may change

A **parameter** marks one quantity as a design variable, something the optimizer is allowed to tune between bounds.
The most common one is the force density of an edge:

```python
from jax_fdm.optimization import EdgeForceDensityParameter


edge = (6, 7)
parameter = EdgeForceDensityParameter(edge, -20.0, -0.1)
```

Each parameter takes an element key and a lower and upper bound (`None` on either side leaves it unbounded).
Here every edge force density is free to move within `[-20.0, -0.1]`, keeping the structure in compression.

We are not limited, however, to force densities.
Support positions (`NodeSupportXParameter`, `NodeSupportZParameter`, …) and applied loads (`NodeLoadXParameter`, …) can be design variables too, and group variants (`EdgeGroupForceDensityParameter`, …) tie many elements to a single shared value.

!!! note "Parameters are optional"

    If we pass `parameters=None`, the optimizer falls back to a sensible default: every edge force density becomes a free variable, unbounded.
    So the quickest constrained run needs no explicit parameters at all, and we reach for them when we want bounds or want to expose supports and loads.

### 2. Goals: what we strive for

A **goal** is a soft target on an quantity in the structure's equilibrium configuration: reach this length, hover over this point, minimize this force.
Soft means the optimizer trades goals off against one another rather than satisfying any single one exactly.

See the guide on [goals](goals.md) for details.
It covers what a goal is and how keys become indices, to spoil you a few things.
Here it is enough to know that JAX FDM ships a large bank of them and that we create one goal per element we want to steer:

```python
from jax_fdm.goals import VertexZCoordinateGoal
from jax_fdm.goals import EdgeForceGoal


vertex = 4
goal_z = VertexZCoordinateGoal(vertex, target=0.5)

edge = (6, 7)
goal_force = EdgeForceGoal(edge, target=2.0)
```

### 3. The loss: how goals become a single number

An optimizer minimizes one scalar.
A **loss** is what turns our list of goals into that scalar, by measuring how far each goal's prediction sits from its target and summing the misfit.

We wrap our goals in an **error term** and hand the term to `Loss`:

```python
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError


goals = [goal_z, goal_force]
loss = Loss(MeanSquaredError(goals=goals, alpha=1.0))
```

`MeanSquaredError` is, for example, is the mean weighted sum of squared prediction-target gaps.
JAX FDM offers a family of error terms for when the squared gap is not what we want:

| Error term | Measures |
| --- | --- |
| `SquaredError` | the summed squared gap to target |
| `MeanSquaredError` | the same, averaged over goals |
| `RootMeanSquaredError` | the root of the mean squared gap, back in the units of the quantity |
| `AbsoluteError`, `MeanAbsoluteError` | the absolute gap, less swayed by outliers |
| `PredictionError`, `MeanPredictionError` | the predicted quantity itself, no target, for minimizing something like load path toward zero |
| `LogMaxError` | a soft one-sided barrier that penalizes only overshoot past the target |

A loss can hold several error terms at once, each with its own `alpha` weight scaling its contribution, so we can, say, chase a target shape while gently minimizing load path.
We can also add a **regularizer** such as `L2Regularizer(alpha)`, which penalizes the force densities themselves to keep the solution well-behaved.

!!! tip "Prediction errors need no target"

    Most error terms compare a prediction against a goal's target.
    `PredictionError` and its mean cousin skip the target and penalize the raw predicted quantity, which is exactly what we want when the aim is to *minimize* a quantity (total load path, total area) rather than drive it to a set value.
    Another way to look at `PredictionError` is as an error term where the target value is `0.0`, which would equivalently push to minimize the predicted quantities.

### 4. Constraints: the lines we will not cross

While a goal is a soft preference, a **constraint** is a hard bound, where the optimizer must keep the constrained quantity between its limits.

```python
from jax_fdm.constraints import EdgeLengthConstraint


constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]
```

Constraints are covered in [constraints](constraints.md).
The fine print here is that they render more complex and potentially more expensive optimization problem, so apply them judiciously if you care about solution speed.
Two things matter when composing a constrained form-finding problem with hard constraints: they are honored only by the optimizers that support them (`SLSQP` and `IPOPT`), and they are optional, leave them out for an unconstrained minimization.

### 5. The optimizer: who does the searching

The **optimizer** is the algorithm that walks the parameter space toward a minimum.
JAX FDM wraps a spread of them, sorted by what they can do:

- **Gradient-based, unconstrained** — `LBFGSB`, `BFGS`, `GradientDescent`, and the second-order Newton and trust-region family (`NewtonCG`, `TruncatedNewton`, `TrustRegionNewton`, …). Fast, and the usual first choice when we have no hard constraints.
- **Gradient-based, constrained** — `SLSQP`, `IPOPT`, `TrustRegionConstrained`. These are the ones that honor the constraints from ingredient 4.
- **Gradient-free** — `DifferentialEvolution`, `DualAnnealing`, `Powell`, `NelderMead`. Slower, but they need no gradients and can escape local minima, useful for rough or non-smooth objectives, or for benchmarking against their gradient-based counterparts.

```python
from jax_fdm.optimization import SLSQP


optimizer = SLSQP()
```

## Putting it together

Here is a complete constrained form-finding run, the same one featured in the [arch optimization](../examples/arch.md) example.
The brief: find the arch that minimizes total load path while keeping every segment between 0.75 and 1 meter.

```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError


length = 5.0
num_segments = 10
segment_length = length / num_segments

xs = [-length / 2.0 + i * segment_length for i in range(num_segments + 1)]
nodes = [[x, 0.0, 0.0] for x in xs]
edges = [(i, i + 1) for i in range(num_segments)]

network = FDNetwork.from_nodes_and_edges(nodes, edges)

network.edges_forcedensities(q=-1.0)
network.nodes_supports(keys=[node for node in network.nodes() if network.is_leaf(node)])
network.nodes_loads([0.0, 0.0, -0.3])

loss = Loss(PredictionError(goals=[NetworkLoadPathGoal()]))
constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]

optimized = constrained_fdm(
    network,
    optimizer=SLSQP(),
    loss=loss,
    constraints=constraints,
)
```

Notice there are no explicit `parameters` here: with `parameters=None` the optimizer treats every edge force density as a free variable, which is exactly what this problem wants.
The `PredictionError` on a `NetworkLoadPathGoal` means "make the load path as small as we can" (no target), while the `EdgeLengthConstraint` list keeps the segments in range.

### Reading the result

`constrained_fdm`, like `fdm`, returns a *copy* of the datastructure in equilibrium, leaving the original untouched.
The optimizer's solution is baked into that copy: every edge reports its optimized force density `q`, along with the resulting `length` and `force`, and every node reports its equilibrium `xyz`, residual, and load.
If you have a sharp eye, the compression-only requirement is handled by construction by the optimization problem, even when we do not set limits on the parameters, since we start from a negative initial guess of `q`, and the path of least resistance for the optimizer in this simple problem is to remain in the compression regime rather than switching to the tension regime (which would result in a steeper optimization landscap and would require a lot more effort).
We read the outcome straight off `optimized` with the usual datastructure accessors, or hand it to the [`Viewer`](../api/jax_fdm.visualization.viewers.md) to see it.

!!! note "Constraints run on the dense solver"

    The sparse solver does not support constraints yet.
    When we pass constraints with the default `sparse=True`, `constrained_fdm` switches the solve to dense and prints a short notice.
    Unconstrained problems keep the sparse solver and its scaling.

## Where to next

- New to the objects `constrained_fdm` juggles under the hood? Start with [form-finding](form_finding.md).
- Need a target or a bound the library does not ship? Write a [custom goal](custom_goals.md) or [custom constraint](custom_constraints.md).
