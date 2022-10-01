<h1 align='center'>JAX FDM</h1>

A differentiable, hardware-accelerated framework for inverse form-finding in structural design.

> Crafted with care in the [Form-Finding Lab](http://formfindinglab.princeton.edu/) at [Princeton University](https://princeton.edu) ❤️🇺🇸

![](images/fdm_header.gif)

JAX FDM enables the solution of inverse form-finding problems for discrete force networks using the force density method (FDM) and gradient-based optimization. It streamlines the integration of form-finding simulations into deep learning models for machine learning research.

## Installation

```bash
pip install jax-fdm
```

Requires Python 3.7+, JAX 0.3.17+, Numpy 1.23.3+, Scipy 1.9.1+, and COMPAS 1.16.0+.
For visualization, use COMPAS_VIEW2 +0.7.0.

## Key features

- **Derivatives, JIT compilation and parallelization.**
JAX FDM is written in [JAX](https://github.com/google/jax), a library for high-performance numerical computing and machine learning research, and it thus inherits many of JAX's perks: calculate derivatives, parallelize, and just-in-time (JIT) compile entire form-finding simulations written in Python code.
<!-- The same JAX code can be run in a CPU, or in multiple GPUs or TPUs (🤯). Accelerate your simulations with minimal burden! -->
- **Legendary form-finding solver.**
JAX FDM computes static equilibrium states for discrete force networks with the [force density method (FDM)](https://www.sciencedirect.com/science/article/pii/0045782574900450), the time-tested form-finding solver backed up by over 50 years of peer-reviewed research.
<!--  -->
- **Autotune those force densities.**
A form-found structure should fulfill additional design requirements to become a feasible structure in the real world.
Formulate an inverse form-finding scenario like this as an optimization problem with JAX FDM.
Then, let one of its gradient-based optimizers solve this readme by automatically tweaking the network's force densities.
<!-- Some popular examples of inverse form-finding problems include best-fitting a vault to an arbitrary target shape, minimizing the load path of a funicular network, or controlling the thrust and the supports of a bridge. -->
<!-- (Coming soon: tweak the support positions and the applied loads, in addition to the force densities, too!). -->
- **A rich bank of goals, constraints and loss functions.**
No two structures are alike.
JAX FDM allows you to model a custom inverse form-finding problem with its (growing!) collection of goals, constraints, and loss functions via a simple, object-oriented API.
The available goals and constraints in the framework are granular and applicable to an entire network; to a subset of its nodes, edges, and combinations thereof.
<!-- Don't see a goal or a constraint you fit?. Add yours with ease! Consult our documentation guide (in progress) to see how you add yours. -->
- **Form-finding simulations as another layer in a neural network.**
Form-finding solver as another layer in a neural network. As an automatically differentiable library, JAX FDM can be seamlessly added as a module of a differentiable function approximator (like a neural network) that can be then trained end-to-end.
Let the neural network learn the underlying physics of static equilibrium *directly* rom a form-finding solver instead of resorting to laborious techniques like data augmentation.
<!--  -->

JAX FDM is a research project under development.
Expect sharp edges and possibly some API breaking changes as we continue to support a broader set of features.

## Documentation

Work in progress! Expect a release soon.

## Quick example

Say the goal is to compute the form of an arch that minimizes the total Maxwell load path of an arch subjected to vertical point loads, while constraining the length of its segments between 0.75 and 1.
We solve this inverse form-finding problem with the SLSQP optimization algorithm.

```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import PredictionError
from jax_fdm.losses import Loss


network = FDNetwork.from_json("data/json/arch.json")
network.edges_forcedensities(q=-1.0)
network.nodes_supports(keys=[node for node in network if network.is_leaf(node)])
network.nodes_loads([0.0, 0.0, -0.2])

goals = [NetworkLoadPathGoal()]
loss = Loss(PredictionError(goals))
constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]
optimizer = SLSQP()

c_network = constrained_fdm(network, optimizer, loss, constraints=constraints)
c_network.to_json("data/json/arch_constrained.json")
```

We visualize the unconstrained form-found `network` (gray) and the constrained one, `c_network` (in teal), with COMPAS VIEW2, and draw lines between their nodes:

![](images/arch_loadpath.png)


## More examples

- [Creased masonry vault](https://github.com/arpastrana/jax_fdm/blob/main/examples/butt.py): Best-fit a target surface [(Panozzo, et al. 2013)](https://cims.nyu.edu/gcl/papers/designing-unreinforced-masonry-models-siggraph-2013-panozzo-et-al.pdf).
- [Pointy dome](https://github.com/arpastrana/jax_fdm/blob/main/examples/dome.py): Control the tilt and the coarse width of a brick dome. 
- [Triple-branching saddle](https://github.com/arpastrana/jax_fdm/blob/main/examples/monkey_saddle.py): Design the distribution of thrusts at the supports of a monkey saddle network while constraining the edge lengths.
- [Saddle bridge](https://github.com/arpastrana/jax_fdm/blob/main/examples/pringle.py): Create a crease in the middle of the bridge while constraining to transversal edges of the network to a target plane. 

## Citation

If you found this library to be useful in academic or industry work, please consider 1) starring the project on Github, and 2) citing it:

```bibtex
@software{pastrana_jaxfdm,
          title={{JAX~FDM}: A differentiable, hardware-accelerated framework for inverse form-finding in structural design},
          author={Rafael Pastrana and Sigrid Adriaenssens},
          year={2022},
          url={https://github.com/arpastrana/jax_fdm}
```

## Acknowledgements

This work has been supported by the **U.S. National Science Foundation** under grant **OAC-2118201** and the [Institute for Data Driven Dynamical Design](https://www.mines.edu/id4/).

## See also

[COMPAS CEM](https://github.com/arpastrana/compas_cem): Inverse design of 3D trusses with the extended Combinatorial Equilibrium Modeling framework.

[JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs.

## License

MIT
