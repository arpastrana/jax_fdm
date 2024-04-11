from time import time

import matplotlib.pyplot as plt

import numpy as np

from jax import vmap

import jax.numpy as jnp
import jax.tree_util as jtu

from jax_fdm import DTYPE_JAX

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import structure_from_datastructure


class LossPlotter:
    """
    Plot a loss function.
    """
    def __init__(self, loss, datastructure, *args, **kwargs):
        self.loss = loss
        self.structure = structure_from_datastructure(datastructure, sparse=False)
        self.fig = plt.figure(**kwargs)

    def plot(self, history, print_breakdown=True, plot_legend=True):
        """
        Plot the loss function and its error components on a list of fdm parameter states.
        """
        print("\nPlotting loss function...")
        start_time = time()

        # Create batched parameter state
        params = jtu.tree_map(lambda leaf: jnp.asarray(leaf, dtype=DTYPE_JAX),
                              history,
                              is_leaf=lambda y: isinstance(y, list))

        # Model is dense because it dense supports vmapping and sparse does not
        model = EquilibriumModel(tmax=1)

        equilibrium_vmap = vmap(model, in_axes=(0, None))
        eq_states = equilibrium_vmap(params, self.structure)

        if print_breakdown:
            print("\n***Error breakdown***")

        errors_all = []

        for error_term in self.loss.terms_error:
            errors = vmap(error_term)(eq_states)
            errors_all.append(errors)
            plt.plot(errors, label=error_term.name)

            if print_breakdown:
                self.print_error_stats(error_term, errors)

        for reg_term in self.loss.terms_regularization:
            errors = vmap(reg_term)(params)
            errors_all.append(errors)
            plt.plot(errors, label=reg_term.name)

            if print_breakdown:
                self.print_error_stats(reg_term, errors)
                print()

        losses = jnp.sum(jnp.asarray(errors_all, dtype=DTYPE_JAX), axis=0)

        plt.plot(losses, label=self.loss.name)

        plt.xlabel("Optimization iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid()

        if plot_legend:
            plt.legend()

        print(f"Plotting time: {(time() - start_time):.4} seconds")

        return losses

    def show(self):
        """
        Display the plot.
        """
        plt.show()

    @staticmethod
    def print_error_stats(error_term, errors):
        """
        Print error statistics
        """
        stats = {"first": errors[0],
                 "last": errors[-1],
                 "min": np.min(errors),
                 "max": np.max(errors)}

        name_string = "{:<18}\t".format(error_term.name)
        values_string = "  ".join(["{}: {:>12.4f}".format(key.capitalize(), value) for key, value in stats.items()])
        print(name_string + values_string)
