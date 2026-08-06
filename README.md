<h1 align='center'>JAX FDM</h1>

<!-- Badges -->
<div align="center">

**Auto-differentiable and hardware-accelerated force density method.**

![build](https://github.com/arpastrana/jax_fdm/actions/workflows/build.yml/badge.svg)
[![docs](https://github.com/arpastrana/jax_fdm/actions/workflows/docs.yml/badge.svg)](https://arpastrana.github.io/jax_fdm/)
[![CMAME](https://img.shields.io/badge/CMAME-10.1016%2Fj.cma.2026.118783-blue.svg)](https://doi.org/10.1016/j.cma.2026.118783)
[![PyPI - Latest Release](https://img.shields.io/pypi/v/jax-fdm.svg)](https://pypi.python.org/project/jax-fdm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jax-fdm.svg)](https://pypi.python.org/project/jax-fdm)
[![arXiv](https://img.shields.io/badge/arXiv-2307.12407-b31b1b.svg)](https://arxiv.org/abs/2307.12407)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7258292-blue.svg)](https://doi.org/10.5281/zenodo.7258292)
[![Made with COMPAS](https://compas.dev/badge.svg)](https://compas.dev/mission-control/#jax_fdm)

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-example">Quick example</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

> Crafted with care at [Princeton University](https://princeton.edu) ❤️🇺🇸

</div>

![](https://raw.githubusercontent.com/arpastrana/jax_fdm/main/docs/assets/images/jax_logo.gif)

<!-- --8<-- [start:pitch] -->
JAX FDM solves inverse design problems for lightweight structures modeled as pin-jointed bar systems using the force density method (FDM) and gradient-based optimization.
It streamlines the integration of mechanical simulations into deep learning models for machine learning research.
<!-- --8<-- [end:pitch] -->

## Features

<!-- --8<-- [start:features] -->
- **Legendary form-finding solver.**
JAX FDM computes static equilibrium states for pin-jointed bar systems with the [force density method (FDM)](https://www.sciencedirect.com/science/article/pii/0045782574900450), the time-tested solver for geometrically nonlinear systems backed up by over 50 years of peer-reviewed research 📚.
<!--  -->
- **Derivatives, JIT compilation, and parallelization.**
JAX FDM is written in [JAX](https://github.com/google/jax), a library for high-performance numerical computing and machine learning research, and it thus inherits many of JAX's perks: calculate derivatives, parallelize, and just-in-time (JIT) compile entire structural simulations written in Python code, and run them on a CPU, a GPU, or a TPU 🤯.
- **Autotune those force densities, loads, and supports.**
A lightweight structure should fulfill additional technical requirements to become a feasible system for real-world construction.
This requires finding the parameters that lead to a specific constrained equilibrium state satisfying these conditions.
Formulate such an inverse problem with JAX FDM, and let one of its gradient-based optimizers solve it by automatically tweaking the system's force densities, applied loads, and support positions 🕺🏻.
- **A rich bank of goals, constraints, and loss functions.**
No two structures are alike.
JAX FDM allows you to model a custom design task with its (growing!) collection of goals, constraints, and loss functions via a simple, object-oriented API.
The available goals and constraints in the framework are granular and applicable to an entire structure; to a subset of its nodes (i.e., vertices), edges, and combinations thereof 💡.
<!-- Don't see a goal or a constraint you fit?. Add yours with ease! Consult our documentation guide (in progress) to see how you add yours. -->
- **Structural simulations as another layer in a neural network.**
As an auto-differentiable library, JAX FDM can be seamlessly added as a layer in a differentiable function approximator like a neural network that can be then trained end-to-end.
Let the neural network learn the underlying physics of static equilibrium *directly* from the simulation, instead of resorting to laborious techniques like data augmentation 🤖.
<!-- --8<-- [end:features] -->

JAX FDM is a research project under development.
Expect sharp edges and possibly some API breaking changes as we continue to support a broader set of features.

## Installation

Install JAX FDM with a one-liner via `pip`:

```bash
pip install jax-fdm
```

This pulls in COMPAS 2.x and the other core dependencies automatically.
JAX FDM supports Python 3.11 to 3.13, and builds on JAX, SciPy, Equinox, and the COMPAS framework.
For the optional extras (3D and notebook viewers, a 2D plotter, the IPOPT optimizer, development tools) and platform notes for Windows, see the [installation guide](https://arpastrana.github.io/jax_fdm/latest/installation/).

## Quick example

Suppose you are interested in [form-finding](https://arpastrana.github.io/jax_fdm/latest/howto/form_finding/) a prestressed cable-net that spans 10 meters per side, a tension-only structure.
You model it as a mesh built from a square grid of 10 by 10 quadrilateral cells, anchor its four corners, and lift two opposite ones by 5 meters.
Then, you set the force density of every cable as the ratio between a target force and a rest length, and compute the cable-net shape with the FDM.

```python
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import fdm


# create a cable-net from a square mesh grid
cablenet = FDMesh.from_meshgrid(10.0, nx=10)

# support the four corners, and lift two opposite ones by 5 meters
corners = list(cablenet.vertices_where(vertex_degree=2))
cablenet.vertices_supports(corners)
cablenet.vertex_attribute(corners[0], "z", 5.0)
cablenet.vertex_attribute(corners[-1], "z", 5.0)

# set force densities from target forces and rest lengths
for edge in cablenet.edges():
    force = 20.0 if cablenet.is_edge_on_boundary(edge) else 1.0
    rest_length = cablenet.edge_length(edge)
    cablenet.edge_forcedensity(edge, force / rest_length)

# form follows function
f_cablenet = fdm(cablenet)
```

![Prestressed cable-net in equilibrium](https://raw.githubusercontent.com/arpastrana/jax_fdm/main/docs/assets/images/cablenet.png)

The net settles into a saddle shape, with all of its cables in tension. Hooray! 🎉
Its boundary cables, however, carry less force than the 20 kN you asked for: a force density prescribes a force *per unit length*, and the cable lengths change as the net finds equilibrium.
Forward simulation is thus tractable, but it leaves your control over such target properties indirect.
Meeting them to design a *buildable* cable-net calls for [constrained form-finding](https://arpastrana.github.io/jax_fdm/latest/howto/constrained_form_finding/), which JAX FDM tackles with gradient-based optimization.
To see constrained form-finding in action, continue this example in the [cable-net prestress guide](https://arpastrana.github.io/jax_fdm/latest/examples/cablenet/), which adds goals for the target force and the target cable length, and optimizes the tension-only shape.
The docs also collect [more example scripts](https://arpastrana.github.io/jax_fdm/latest/examples/more/) and runnable Colab notebooks.

## Documentation

The full documentation lives at [arpastrana.github.io/jax_fdm](https://arpastrana.github.io/jax_fdm/).

## Citation

If you found this library to be useful in academic or industry work, please consider (1) starring the project on Github, and (2) citing it:


``` bibtex
@article{pastrana_dfdm_2026,
         title = {Differentiable force density method for the design of lightweight structures},
         author = {Pastrana, Rafael and Oktay, Deniz and Bletzinger, Kai-Uwe and Adams, Ryan P. and Adriaenssens, Sigrid},
         date = {2026},
         journaltitle = {Computer Methods in Applied Mechanics and Engineering},
         volume = {458},
         pages = {118783},
         issn = {00457825},
         doi = {10.1016/j.cma.2026.118783}}
```

``` bibtex
@inproceedings{pastrana_jaxfdm_2023,
               title = {{{JAX FDM}}: {{A}} differentiable solver for inverse form-Finding},
               booktitle = {Differentiable {{Almost Everything Workshop}} of the 40th {{International Conference}} on {{Machine Learning}}},
               author = {Pastrana, Rafael and Oktay, Deniz and Adams, Ryan P. and Adriaenssens, Sigrid},
               year = {2023},
               address = {Hawaii, USA},
               url = {https://openreview.net/forum?id=Uu9OPgh24d}}
```

## Acknowledgements

This work has been supported by the **U.S. National Science Foundation** under grant **OAC-2118201** and the [Institute for Data Driven Dynamical Design](https://www.mines.edu/id4/).

## See also

[COMPAS CEM](https://github.com/arpastrana/compas_cem): Inverse design of 3D trusses with the combinatorial equilibrium modeling (CEM) framework.

[JAX CEM](https://github.com/arpastrana/jax_cem): The combinatorial equilibrium modeling (CEM) framework in JAX.

[JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs.

## License

MIT
