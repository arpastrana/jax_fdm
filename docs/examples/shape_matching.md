# Shape Matching

The [arch optimization](arch.md) example minimized a mechanical quantity, the load path.
This time the design intent is *visual*: you have a shape in mind, and you want a structure that looks like it while still standing up under its own weight in pure compression.

That is the shape-matching problem, and it sits at a happy intersection.
A funicular geometry, one that carries its loads through axial forces alone, is mechanically efficient: it is the shape a hanging net or a masonry vault naturally wants to take.
But left to its own devices, form-finding gives you *a* funicular shape, not necessarily *the* one your design calls for.
Shape matching lets you steer the force density method toward a target surface, so mechanical efficiency and your visual intent meet in the same geometry.[^cmame]
It is how you design compression-only gridshells and thin shells that keep their intended silhouette, and how you assess whether an existing masonry vault can stand as a thrust network.

In this walkthrough you approximate a doubly-curved **vault** with a compression-only network.
The recipe is the same one that scales to any target: pick a shape, form-find a first guess, then let the optimizer find the force densities that best fit the target.

## The ingredients: an initial network and a target

Shape matching needs two datastructures: the **network you form-find**, and the **target** whose vertex positions you aim for.
Both ship with the library.
The initial network `vault0` is flat, a plan grid of the vault's footprint; the target `vault` is the double-curved surface you want to approximate.

```python
from jax_fdm.datastructures import FDNetwork


network = FDNetwork.from_json("data/json/vault0.json")
network_target = FDNetwork.from_json("data/json/vault.json")
```

They share the same 193 nodes and 324 edges, node for node, so a node in the flat grid has a matching node on the target surface.
That correspondence is what makes "match the target" a well-posed goal: each free node knows exactly which point it should reach.

You anchor the network along its perimeter (the leaf nodes), hang a small downward point load on every free node, and seed every edge with a constant force density, the same starting guess for all of them.

```python
anchors = [node for node in network.nodes() if network.is_leaf(node)]
network.nodes_anchors(anchors)
network.nodes_loads([0.0, 0.0, -0.2], keys=network.nodes_free())
network.edges_forcedensities(q=-2.0)
```

The negative force density puts every edge in compression, which is the whole point: you want a shape that stands in pure compression, like an arch or a vault.

## Before: a first form-found guess

What does that constant force density give you on its own?
Run one plain form-finding pass and see.

```python
from jax_fdm.equilibrium import fdm


network_guess = fdm(network)
```

![Shape matching, initial guess](../assets/images/shape_matching_initial.png)

The result is funicular, it hangs in equilibrium, but it is a poor likeness of the target.
With a single force density shared across every edge, the network sags into a shallow mound that rises to only about 2.1 meters, while the target vault peaks near 4.5.
Measured as a [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance), the largest gap between the guess and the target is about 5.6 meters.
A constant force density simply cannot bend the shape to match a doubly-curved surface; the force densities need to *vary* across the network, and finding that variation by hand is hopeless.
That is a job for optimization.

## Defining the shape-matching problem

You want the force densities that pull the form-found network as close to the target as possible.
Three pieces express that intent.

**The parameters** are the force densities of every edge, free to move between a lower and an upper bound.
Keeping both bounds negative holds the whole structure in compression throughout the search.

```python
from jax_fdm.parameters import EdgeForceDensityParameter


parameters = [EdgeForceDensityParameter(edge, -20.0, 0.0) for edge in network.edges()]
```

**The goals** ask each free node to reach its counterpart on the target.
You create one [`NodePointGoal`](../api/jax_fdm.goals.md) per free node, aimed at the target's coordinates for that node.

```python
from jax_fdm.goals import NodePointGoal


goals = []
for node in network.nodes_free():
    target_xyz = network_target.node_coordinates(node)
    goals.append(NodePointGoal(node, target=target_xyz))
```

**The loss** measures how far, on average, the nodes land from their targets.
The root-mean-squared error reports that gap in the units of the problem, meters, which makes it easy to reason about.

```python
from jax_fdm.losses import Loss
from jax_fdm.losses import RootMeanSquaredError


loss = Loss(RootMeanSquaredError(goals))
```

Minimizing this loss is exactly the least-squares match the design calls for: an exact fit would drive it to zero, but for an arbitrary target that is unlikely, since it would mean the target was already funicular.
The optimizer instead finds the funicular geometry that comes closest.

## After: the optimized match

Hand the problem to `constrained_fdm` with a gradient-based optimizer.
`LBFGSB` handles the bounded force densities well and converges in a fraction of a second.

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import LBFGSB


network_matched = constrained_fdm(
    network,
    optimizer=LBFGSB(),
    loss=loss,
    parameters=parameters,
    maxiter=1000,
    tol=1e-6,
)
```

![Shape matching, optimized result](../assets/images/shape_matching_optimized.png)

The optimized network rises to about 4.46 meters, all but reaching the target's 4.49, and the Hausdorff distance drops from 5.6 meters to 0.69, an order of magnitude closer.
The force densities now follow a non-trivial distribution across the edges, exactly the spread a single constant value could never provide, and every one of them stayed negative, so the match is compression-only as intended.

!!! tip "Why the match matters mechanically"

    The payoff is not only visual. A shape close to funicular carries its loads mostly through axial forces rather than bending, which is what makes shells and gridshells so material-efficient. Nudging an arbitrary target toward its nearest compression-only geometry can cut its strain energy substantially, so a small geometric change buys an outsized gain in structural performance.

## Reading the result

You can measure the quality of the fit directly, the same way the numbers above were computed, with a Hausdorff distance between the two point sets:

```python
import numpy as np
from scipy.spatial.distance import directed_hausdorff


matched = np.array([network_matched.node_coordinates(n) for n in network_matched.nodes()])
target = np.array([network_target.node_coordinates(n) for n in network_target.nodes()])

hausdorff = max(directed_hausdorff(matched, target)[0], directed_hausdorff(target, matched)[0])
print(f"Hausdorff distance: {hausdorff:.3f}")
```

To see the match, draw the optimized network colored by its force densities next to the target as a plain wireframe, with a line from each node to the target point it was chasing.

```python
from compas.datastructures import Network
from compas.colors import Color
from compas.geometry import Line
from jax_fdm.visualization import Viewer


viewer = Viewer()
viewer.add(network_matched, edgewidth=(0.1, 0.3), edgecolor="fd", loadscale=5.0)
viewer.add(network_target.copy(cls=Network), show_points=False, color=Color.grey())

for node in network_matched.nodes():
    line = Line(network_matched.node_coordinates(node), network_target.node_coordinates(node))
    viewer.add(line, color=Color.red())

viewer.show()
```

The red lines, so prominent in the initial guess, shrink to almost nothing after optimization, a visual echo of the collapsed Hausdorff distance.

## Where to next

- Curious how the loss, goals, and optimizer fit together? Read [constrained form-finding](../howto/constrained_form_finding.md).
- Want a target the goal bank does not cover? Write a [custom goal](../howto/custom_goals.md).
- The runnable script for this example lives in [`examples/vault/vault.py`](https://github.com/arpastrana/jax_fdm/blob/main/examples/vault/vault.py).

[^cmame]:
    The shape-matching task, and the strain-energy analysis behind the efficiency claims, are studied in detail in Pastrana et al., *Differentiable force density method for the design of lightweight structures*, Computer Methods in Applied Mechanics and Engineering (2026). See the [citation page](../citation.md).
