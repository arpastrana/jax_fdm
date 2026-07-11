# Equilibrium

The heart of JAX FDM: compute states of static equilibrium with the force
density method, differentiably.

## Form-finding

The two entry points. `fdm` computes an equilibrium state for fixed parameters;
`constrained_fdm` searches for the parameters whose equilibrium state minimizes
a loss function, subject to constraints.

::: jax_fdm.equilibrium.fdm.fdm
    options:
      heading_level: 3

::: jax_fdm.equilibrium.fdm.constrained_fdm
    options:
      heading_level: 3

---

## Models

::: jax_fdm.equilibrium.models.EquilibriumModel
    options:
      heading_level: 3

::: jax_fdm.equilibrium.models.EquilibriumModelSparse
    options:
      heading_level: 3

---

## States

::: jax_fdm.equilibrium.states.EquilibriumState
    options:
      heading_level: 3

::: jax_fdm.equilibrium.states.LoadState
    options:
      heading_level: 3

::: jax_fdm.equilibrium.states.EquilibriumParametersState
    options:
      heading_level: 3

---

## Structures

The static topology and indexing of a structure, precomputed once so that the
equilibrium model can run shape updates as pure array operations.

::: jax_fdm.equilibrium.structures.structures.EquilibriumStructure
    options:
      heading_level: 3

::: jax_fdm.equilibrium.structures.structures.EquilibriumStructureSparse
    options:
      heading_level: 3

::: jax_fdm.equilibrium.structures.structures.EquilibriumMeshStructure
    options:
      heading_level: 3

::: jax_fdm.equilibrium.structures.structures.EquilibriumMeshStructureSparse
    options:
      heading_level: 3

::: jax_fdm.equilibrium.structures.graphs.Graph
    options:
      heading_level: 3

::: jax_fdm.equilibrium.structures.graphs.GraphSparse
    options:
      heading_level: 3

---

## Iterative solvers

Solver functions for equilibrium states with shape-dependent loads, passed to
`fdm` and `constrained_fdm` via `itersolve_fn`.

::: jax_fdm.equilibrium.solvers.fixed_point.solver_forward
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.fixed_point.solver_fixedpoint
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.fixed_point.solver_anderson
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.root_finding.solver_newton
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.least_squares.solver_gauss_newton
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.least_squares.solver_levenberg_marquardt
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.least_squares.solver_dogleg
    options:
      heading_level: 3

---

## Implicit differentiation

The `custom_vjp`-wrapped solver functions that backpropagate through an
equilibrium state via the implicit function theorem instead of unrolling
solver iterations.

::: jax_fdm.equilibrium.solvers.fixed_point.solver_fixedpoint_implicit
    options:
      heading_level: 3

::: jax_fdm.equilibrium.solvers.nonlinear.solver_nonlinear_implicit
    options:
      heading_level: 3

---

## Sparse solver

::: jax_fdm.equilibrium.sparse.sparse_solve
    options:
      heading_level: 3

::: jax_fdm.equilibrium.sparse.register_sparse_solver
    options:
      heading_level: 3
