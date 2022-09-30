<h1 align='center'>JAX FDM</h1>

A differentiable, hardware-accelerated framework for inverse form-finding.

![](fdm_header.gif)

JAX FDM enables the solution of inverse form-finding problems of discrete force networks, and streamlines the integration of form-fiding simulations into deep learning architectures for machine learnign research. 


- simple to use API.
- solutions via gradient based optimization
- bank of functions
- force density method, time tested, legendary.
- one code 

JAX FDM accelerated their solution streamlines their solution via gradient-based optimization.

This tool offers a (growing!) collection of algotiths, loss functions, goals, and constraints that you can mix and match freely to model a custom constrained form-fiding problem. 

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
import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.numpy.ndarray
    bias: jax.numpy.ndarray

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias
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

## See also

[COMPAS CEM](https://github.com/arpastrana/compas_cem): Inverse design of 3D trusses with the extended Combinatorial Equilibrium Modeling framework.
[JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more.
