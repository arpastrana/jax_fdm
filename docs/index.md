# Overview

Lightweight structures span long distances with slender cross-sections due to their mechanically efficient shapes.
However, simulating these structures and turning them into feasible designs that satisfy technical constraints remains challenging due to geometrically nonlinear mechanical behaviors and high-dimensional search spaces.

JAX FDM enables the solution of inverse problems for lightweight structures modeled as pin-jointed bar systems using the force density method (FDM) and gradient-based optimization.
It streamlines the integration of mechanical simulations into deep learning models for machine learning research.

![JAX FDM](assets/images/jax_logo.gif)

## Key features

- **Legendary form-finding solver.**
JAX FDM computes static equilibrium states for pin-jointed bar systems with the [force density method (FDM)](https://www.sciencedirect.com/science/article/pii/0045782574900450), the time-tested solver for geometrically nonlinear systems backed up by over 50 years of peer-reviewed research 📚.
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
- **Structural simulations as another layer in a neural network.**
As an auto-differentiable library, JAX FDM can be seamlessly added as a layer in a differentiable function approximator like a neural network that can be then trained end-to-end.
Let the neural network learn the underlying physics of static equilibrium *directly* from the simulation, instead of resorting to laborious techniques like data augmentation 🤖.

!!! warning "Work in progress"

    JAX FDM is a research project under development.
    Expect sharp edges and possibly some API breaking changes as we continue to support a broader set of features.

## Acknowledgements

This work has been supported by the **U.S. National Science Foundation** under grant **OAC-2118201** and the [Institute for Data Driven Dynamical Design](https://www.mines.edu/id4/).

## See also

- [COMPAS CEM](https://github.com/arpastrana/compas_cem): Inverse design of 3D trusses with the combinatorial equilibrium modeling (CEM) framework.
- [JAX CEM](https://github.com/arpastrana/jax_cem): The combinatorial equilibrium modeling (CEM) framework in JAX.
- [JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs.
