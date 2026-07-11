# Installation

Install JAX FDM with a one-liner via `pip`:

```bash
pip install jax-fdm
```

This pulls in COMPAS 2.x and the other core dependencies automatically.
JAX FDM supports Python 3.10 to 3.12, and builds on JAX, SciPy, Equinox, and the COMPAS framework.
See [`pyproject.toml`](https://github.com/arpastrana/jax_fdm/blob/main/pyproject.toml) for the complete dependency list.

## Optional extras

JAX FDM declares optional dependency groups you can install from a source checkout with `pip`:

```bash
pip install -e ".[viz]"    # 3D desktop viewer (compas_viewer) and notebook viewer (compas_notebook)
pip install -e ".[ipopt]"  # the IPOPT interior-point optimizer (cyipopt)
pip install -e ".[dev]"    # development tools (ruff, pytest, pre-commit, build, bump-my-version)
```

The `ipopt` extra needs a system Ipopt library available on your machine.
In a Jupyter notebook, use `from jax_fdm.visualization import NotebookViewer` to display structures inline.

For 2D matplotlib plots, install the standalone [compas_plotters](https://github.com/compas-dev/compas_plotters) (not on PyPI yet, hence not part of the `viz` extra):

```bash
pip install git+https://github.com/compas-dev/compas_plotters
```

## Are you a Windows user?

JAX now provides official native CPU wheels for Windows, so JAX FDM should work directly.
On Windows you may also need to install the [Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

For GPU acceleration on Windows, native support is unavailable.
You can instead run JAX through the [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about), but keep in mind that it has no graphical output and that support for this configuration is experimental.
Please refer to [JAX's installation instructions](https://docs.jax.dev/en/latest/installation.html) for details.
