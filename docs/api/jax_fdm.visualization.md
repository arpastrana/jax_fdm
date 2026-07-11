# Visualization

Draw force density datastructures — edges colored by force or force density,
support markers, load and reaction arrows — in a 3D desktop viewer, a Jupyter
notebook, or a 2D matplotlib plot.

!!! note

    The viewers and plotters need optional dependencies: install the `viz`
    extra for the 3D and notebook viewers, and the standalone
    [compas_plotters](https://github.com/compas-dev/compas_plotters) for the
    2D plotter. Without them, the corresponding class degrades to a null
    object that warns on use.

Pick a backend:

- **[3D viewer](jax_fdm.visualization.viewers.md)** — an interactive desktop
  viewer with a scene tree, element selection, and animation support.
- **[Notebook viewer](jax_fdm.visualization.notebooks.md)** — a draw-once
  pythreejs widget for Jupyter notebooks and Colab.
- **[2D plotter](jax_fdm.visualization.plotters.md)** — matplotlib drawings
  for figures and papers, plus a plotter for loss histories.
