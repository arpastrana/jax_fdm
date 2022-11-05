import numpy as np
import matplotlib.pyplot as plt

from jax_fdm import DTYPE_NP
from jax_fdm.equilibrium import EquilibriumModel


__all__ = ["LossPlotter"]


class LossPlotter:
    """
    Plot a loss function.
    """
    def __init__(self, loss, network, *args, **kwargs):
        self.loss = loss
        self.network = network
        self.model = EquilibriumModel(network)
        self.fig = plt.figure(**kwargs)

    def plot(self, qs):
        """
        Plot the loss function on an array of force densities.
        """
        xyz_fixed = np.asarray([self.network.node_coordinates(node) for node in self.network.nodes_fixed()], dtype=DTYPE_NP)
        loads = np.asarray(list(self.network.nodes_loads()), dtype=DTYPE_NP)

        for loss_term in [self.loss] + list(self.loss.terms):
            errors = []
            for q in qs:
                eqstate = self.model(q, xyz_fixed, loads)
                try:
                    error = loss_term(eqstate)
                except TypeError:
                    error = loss_term(q, xyz_fixed, loads, self.model)
                errors.append(error)
            plt.plot(errors, label=loss_term.name)

        plt.xlabel("Optimization iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.show()

    def show(self):
        """
        Display the plot.
        """
        plt.show()
