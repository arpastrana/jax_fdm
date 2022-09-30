<h1 align='center'>JAX FDM</h1>

A differentiable, hardware-accelerated framework for inverse form-finding.

![](fdm_header.gif)

JAX FDM enables the solution of inverse form-finding problems of discrete force networks and streamlines the integration of form-fiding simulations into deep learning architectures for machine learning research. 


- simple to use API.
- solutions via gradient based optimization
- bank of functions
- force density method, time tested, legendary.
- one code 

JAX FDM accelerated their solution streamlines their solution via gradient-based optimization.

This tool offers a (growing!) collection of algotiths, loss functions, goals, and constraints that you can mix and match freely to model a custom constrained form-fiding problem. 
goals and constraints at the node, edge and network level.

the time-tested, Force Density method. Tension or compression only

Code written in JAX can be run in a CPU, a GPU or a TPU with marginal alteration, which streamlines reaseach
machine learning.

## Installation

```bash
pip install jax-fdm
```

Requires Python 3.7+, JAX 0.3.17+, Numpy 1.23.3+, Scipy 1.9.1+, and COMPAS 1.16.0+.
For visualization, we use COMPAS_VIEW2 +0.7.0.

## Documentation

Work in progress! Expect a release soon.

## Quick example


```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import PredictionError
from jax_fdm.losses import Loss


network = Network.from_json("data/arch.json")

loss = Loss(PredictionError(goals=[NetworkLoadPath()]))
constraints = [EdgeLengthConstraint(edge, 1.0, 4,5) for edge in network.edges()]
optimizer = SLSQP()

c_network = constrained_fdm(network, optimizer, loss, constraints)
c_network.to_json("data/arch_constrained.json")
```

## More examples

- Butt
- Pringle
- Monkey saddle
- Dome

## Citation

If you found this library to be useful in academic or industry work, please consider 1) starring the project on Github, and 2) citing it:

```bibtex
@software{pastrana_jaxfdm,
    title={{JAX~FDM}: A differentiable, hardware-accelerated framework for inverse form-finding},
    author={Rafael Pastrana and Sigrid Adriaenssens}
    year={2022},
    url={https://github.com/arpastrana/jax_fdm}
```

## Acknowledgements

This work has been supported by the **U.S. National Science Foundation** under grant **OAC-2118201** and the [Institute for Data Driven Dynamical Design](https://www.mines.edu/id4/).


## See also

[COMPAS CEM](https://github.com/arpastrana/compas_cem): Inverse design of 3D trusses with the extended Combinatorial Equilibrium Modeling framework.
[JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more.
