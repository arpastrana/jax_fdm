import matplotlib.pyplot as plt
import numpy as np
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
        for loss_term in [self.loss] + list(self.loss.terms):
            errors = []
            for q in qs:
                eqstate = self.model(q)
                try:
                    error = loss_term(eqstate)
                except TypeError:
                    error = loss_term(q, self.model)
                errors.append(error)
            plt.plot(errors, label=loss_term.name)

        plt.xlabel("Optimization iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid()
        plt.legend()

    def show(self):
        """
        Display the plot.
        """
        plt.show()

    def save(self, qs, filepath):
        """
        Save the plot.
        """
        # ax.set_xlabel("Optimization iterations")
        # ax.set_ylabel("Loss value")
        # ax.grid(which="major")

        losses = [self.loss(np.array(q), self.model) for q in qs]
        x = np.arange(len(qs))

        for i in range(len(qs)):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
            ax.clear()
            ax.set_ylim(0, 4)
            ax.set_xlim(0, 150)

            ax.set_xlabel("Optimization iterations")
            ax.set_ylabel("Loss value")
            ax.grid(which="both")

            line, = ax.plot(x[0:i], losses[0:i], lw=2, color="tab:orange")
            point, = ax.plot(x[i], losses[i], marker='.', color='tab:orange')

            plt.savefig(f"{filepath}/{i}.png")
            plt.close()
