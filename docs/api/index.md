# API reference

The public API of JAX FDM, organized by subpackage. A typical workflow touches
them in this order:

1. **[datastructures](jax_fdm.datastructures.md)** — model a structure as an
   `FDNetwork` or an `FDMesh`, and set its force densities, supports, and loads.
2. **[equilibrium](jax_fdm.equilibrium.md)** — compute a state of static
   equilibrium with `fdm`, or solve an inverse problem with `constrained_fdm`.
3. **[goals](jax_fdm.goals.md)** — describe the target properties of the
   equilibrium state you are after.
4. **[constraints](jax_fdm.constraints.md)** — bound quantities of the
   equilibrium state during constrained optimization.
5. **[losses](jax_fdm.losses.md)** — combine goals into the error function that
   an optimizer minimizes.
6. **[optimization](jax_fdm.optimization.md)** — the gradient-based and
   gradient-free optimizers that solve the inverse problem.
7. **[parameters](jax_fdm.parameters.md)** — choose which quantities of the
   structure the optimizer is allowed to tweak.
8. **[geometry](jax_fdm.geometry.md)** — differentiable geometric primitives
   used by goals and constraints.
9. **[visualization](jax_fdm.visualization.md)** — draw force density
   datastructures in a 3D viewer, a Jupyter notebook, or a 2D plot.
