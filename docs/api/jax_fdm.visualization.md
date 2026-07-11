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

## 3D viewer

::: jax_fdm.visualization.viewers.viewer.Viewer
    options:
      heading_level: 3

::: jax_fdm.visualization.viewers.scene_objects.FDNetworkObject
    options:
      heading_level: 3

::: jax_fdm.visualization.viewers.scene_objects.FDMeshObject
    options:
      heading_level: 3

---

## Notebook viewer

::: jax_fdm.visualization.notebooks.viewer.NotebookViewer
    options:
      heading_level: 3

::: jax_fdm.visualization.notebooks.scene_objects.ThreeFDNetworkObject
    options:
      heading_level: 3

::: jax_fdm.visualization.notebooks.scene_objects.ThreeFDMeshObject
    options:
      heading_level: 3

---

## 2D plotter

::: jax_fdm.visualization.plotters.plotter.Plotter
    options:
      heading_level: 3

::: jax_fdm.visualization.plotters.scene_objects.FDNetworkPlotterObject
    options:
      heading_level: 3

::: jax_fdm.visualization.plotters.scene_objects.FDMeshPlotterObject
    options:
      heading_level: 3

::: jax_fdm.visualization.plotters.loss_plotter.LossPlotter
    options:
      heading_level: 3
