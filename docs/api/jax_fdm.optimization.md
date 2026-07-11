# Optimization

The optimizers that solve constrained form-finding problems, plus recorders to
replay an optimization history.

## Base optimizers

::: jax_fdm.optimization.optimizers.optimizer.Optimizer
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.constrained.ConstrainedOptimizer
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_free.GradientFreeOptimizer
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.second_order.SecondOrderOptimizer
    options:
      heading_level: 3

---

## Gradient-based optimizers

::: jax_fdm.optimization.optimizers.gradient_descent.GradientDescent
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.BFGS
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.LBFGSB
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.LBFGSBS
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.SLSQP
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.TruncatedNewton
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.TrustRegionConstrained
    options:
      heading_level: 3

---

## Second-order optimizers

::: jax_fdm.optimization.optimizers.gradient_based.NewtonCG
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.TrustRegionExact
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.TrustRegionKrylov
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_based.TrustRegionNewton
    options:
      heading_level: 3

---

## Interior-point optimizers

Requires the `ipopt` extra.

::: jax_fdm.optimization.optimizers.ipopt.IPOPT
    options:
      heading_level: 3

---

## Gradient-free optimizers

::: jax_fdm.optimization.optimizers.gradient_free.NelderMead
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.gradient_free.Powell
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.evolutionary.DifferentialEvolution
    options:
      heading_level: 3

::: jax_fdm.optimization.optimizers.evolutionary.DualAnnealing
    options:
      heading_level: 3

---

## Recording

::: jax_fdm.optimization
    options:
      heading_level: 3
      members:
        - OptimizationRecorder
        - Collection
