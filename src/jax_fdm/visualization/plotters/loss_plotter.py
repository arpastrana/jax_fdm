from collections.abc import Iterable
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import structure_from_datastructure
from jax_fdm.losses import Loss


class LossPlotter:
    """
    Plot a loss function.
    """
    def __init__(self, loss: Loss, datastructure: FDNetwork | FDMesh, **kwargs: Any) -> None:
        self.loss = loss
        self.structure = structure_from_datastructure(datastructure, sparse=False)
        self.fig = plt.figure(**kwargs)

    def plot(
        self,
        history: list,
        report_breakdown: bool = True,
        error_names: Iterable[str] | None = None,
        plot_legend: bool = True,
        yscale: str = "log",
        **eq_kwargs: Any,
    ) -> jax.Array:
        """
        Plot the loss function and its error components on a list of fdm parameter states.
        """
        print("\nPlotting loss function...")
        start_time = perf_counter()

        # Create batched parameter state
        params = jtu.tree_map(lambda leaf: jnp.asarray(leaf, dtype=DTYPE_JAX),
                              history,
                              is_leaf=lambda y: isinstance(y, list))

        if not eq_kwargs:
            eq_kwargs = {"tmax": 1}

        # Model is dense because it dense supports vmapping and sparse does not
        model = EquilibriumModel(**eq_kwargs)

        equilibrium_vmap = vmap(model, in_axes=(0, None))
        eq_states = equilibrium_vmap(params, self.structure)

        # Calculate error and regularization contributions
        errors_all: dict[str, jax.Array] = {}

        for error_term in self.loss.terms_error:
            errors = vmap(error_term)(eq_states)
            errors_all[error_term.name] = errors

        for reg_term in self.loss.terms_regularization:
            errors = vmap(reg_term)(params)  # pyright: ignore[reportArgumentType]  # Regularizer defines __call__ only on its subclasses, not on the base class
            errors_all[reg_term.name] = errors

        # Plot loss
        losses = jnp.sum(jnp.asarray(list(errors_all.values()), dtype=DTYPE_JAX), axis=0)
        self.print_error_stats(losses, "Loss")
        plt.plot(losses, label=self.loss.name)

        # Report loss breakdown
        if report_breakdown:
            print("\n***Error breakdown***")
            if error_names is None:
                error_names = errors_all.keys()

            for name in error_names:
                errors = errors_all[name]
                plt.plot(errors, label=name)
                self.print_error_stats(errors, name)

        plt.xlabel("Optimization iterations")
        plt.ylabel("Loss")
        plt.yscale(yscale)
        plt.grid()

        if plot_legend:
            plt.legend()

        print(f"Plotting time: {(perf_counter() - start_time):.4} seconds")

        return losses

    def show(self) -> None:
        """
        Display the plot.
        """
        plt.show()

    @staticmethod
    def print_error_stats(errors: jax.Array | np.ndarray, error_name: str) -> None:
        """
        Print error statistics
        """
        stats = {"first": errors[0],
                 "last": errors[-1],
                 "min": np.min(errors),
                 "max": np.max(errors)}

        name_string = "{:<18}\t".format(error_name)
        values_string = "  ".join(["{}: {:>12.6f}".format(key.capitalize(), value) for key, value in stats.items()])
        print(name_string + values_string)
