import matplotlib.pyplot as plt

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
        plt.show()

    def show(self):
        """
        Display the plot.
        """
        plt.show()
