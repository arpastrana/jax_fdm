from time import time

import matplotlib.pyplot as plt

from jax import vmap
import jax.numpy as jnp

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumStructure


__all__ = ["LossPlotter"]


class LossPlotter:
    """
    Plot a loss function.
    """
    def __init__(self, loss, network, *args, **kwargs):
        self.loss = loss
        self.structure = EquilibriumStructure.from_network(network)
        self.fig = plt.figure(**kwargs)

    def plot(self, history):
        """
        Plot the loss function and its error components on a list of fdm parameter arrays.
        """
        print("\nPlotting loss function...")
        start_time = time()

        q = jnp.asarray(history["q"], dtype=DTYPE_JAX)
        xyz_fixed = jnp.asarray(history["xyz_fixed"], dtype=DTYPE_JAX)
        loads = jnp.asarray(history["loads"], dtype=DTYPE_JAX)

        model = EquilibriumModel(q, xyz_fixed, loads) # NOTE: dummy parameters
        equilibrium_vmap = vmap(model.equilibrium, in_axes=(0, 0, 0, None))
        eq_states = equilibrium_vmap(q, xyz_fixed, loads, self.structure)

        errors_all = []
        for error_term in self.loss.terms:
            errors = vmap(error_term)(eq_states)
            errors_all.append(errors)
            plt.plot(errors, label=error_term.name)

        losses = jnp.sum(jnp.asarray(errors_all, dtype=DTYPE_JAX), axis=0)
        plt.plot(losses, label=self.loss.name)

        plt.xlabel("Optimization iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        print(f"Plotting time: {(time() - start_time):.4} seconds")

    def show(self):
        """
        Display the plot.
        """
        plt.show()
