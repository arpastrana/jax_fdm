# Arch Optimization

Suppose you are interested in finding a suitable funicular geometry for a 10-meter span arch subjected to vertical point loads of 0.3 kN.
The arch has to be compression-dominant.
You model the arch as a `jax_fdm` network (download the arch `json` file [here](https://github.com/arpastrana/jax_fdm/blob/main/data/json/arch.json)).
Then, you apply a force density of -1 to all of its edges, and compute the required shape with the force density method.

```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm


network = FDNetwork.from_json("data/json/arch.json")
network.edges_forcedensities(q=-1.0)
network.nodes_supports(keys=[node for node in network.nodes() if network.is_leaf(node)])
network.nodes_loads([0.0, 0.0, -0.3])

f_network = fdm(network)
```

You now wish to find a new form for this arch that minimizes the [total Michell's load path](https://doi.org/10.1007/s00158-019-02214-w), while keeping the length of the arch segments between 0.75 and 1 meters.
You solve this constrained form-finding problem with the SLSQP gradient-based optimizer.

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import PredictionError
from jax_fdm.losses import Loss


loss = Loss(PredictionError(goals=[NetworkLoadPathGoal()]))
constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]
optimizer = SLSQP()

c_network = constrained_fdm(network, optimizer, loss, constraints=constraints)
```

You finally visualize the constrained arch `c_network` with the `Viewer`, together with the unconstrained arch `f_network` as a plain wireframe (convert it to a COMPAS `Network` to draw it without the force density styling).

```python
from compas.datastructures import Network
from jax_fdm.visualization import Viewer


viewer = Viewer(width=1600, height=900)
viewer.add(c_network)
viewer.add(f_network.copy(cls=Network))
viewer.show()
```

![Arch load path](../assets/images/arch_loadpath.png)

The constrained form is shallower than the unconstrained one as a result of the optimization process.
The length of the arch segments also varies within the prescribed bounds to minimize the load path: segments are the longest where the arch's internal forces are lower (1.0 meter, at the apex); and conversely, the segments are shorter where the arch's internal forces are higher (0.75 m, at the base).
