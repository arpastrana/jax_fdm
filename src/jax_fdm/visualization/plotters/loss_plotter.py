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
        self.fig = plt.figure(**kwargs)

    def plot(self, xs):
        """
        Plot the loss function on an array of force densities.
        """
        model = EquilibriumModel(self.network)

        for loss_term in [self.loss] + list(self.loss.terms):
            errors = []
            for x in xs:
                q, xyz_fixed, loads = x
                eqstate = model(q, xyz_fixed, loads)
                try:
                    error = loss_term(eqstate)
                except TypeError:
                    error = loss_term(q, xyz_fixed, loads, model)
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
