# jax_fdm.losses

A loss function scalarizes the deviation between the current equilibrium state
and the [goals](jax_fdm.goals.md) into the objective that an
[optimizer](jax_fdm.optimization.md) minimizes.

## Loss

::: jax_fdm.losses
    options:
      heading_level: 3
      members:
        - Loss

---

## Error terms

::: jax_fdm.losses
    options:
      heading_level: 3
      members:
        - Error
        - PredictionError
        - MeanPredictionError
        - SquaredError
        - MeanSquaredError
        - RootMeanSquaredError
        - AbsoluteError
        - MeanAbsoluteError
        - LogMaxError

---

## Regularizers

::: jax_fdm.losses
    options:
      heading_level: 3
      members:
        - Regularizer
        - L2Regularizer
