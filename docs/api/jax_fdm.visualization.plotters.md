# Plotters

Vectorized drawings of force density datastructures for figures and papers.
The 2D plotter is built on the standalone
[compas_plotter](https://pypi.org/project/compas-plotter/). A second
plotter, built with [matplotlib](https://matplotlib.org), charts the loss
histories recorded during optimization.

!!! note

    The plotter needs the standalone [compas_plotter](https://pypi.org/project/compas-plotter/) for use.
    Without it, the plotter degrades to a null object that warns on use.


::: jax_fdm.visualization.plotters.plotter.Plotter
    options:
      heading_level: 2


::: jax_fdm.visualization.plotters.loss_plotter.LossPlotter
    options:
      heading_level: 2
