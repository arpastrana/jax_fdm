# Installation

Install JAX FDM with a one-liner via `pip`:

```bash
pip install jax-fdm
```

This pulls in COMPAS 2.x and the other core dependencies automatically.
JAX FDM supports Python 3.10 to 3.12, and builds on JAX, SciPy, Equinox, and the COMPAS framework.
See the complete [dependency list](https://github.com/arpastrana/jax_fdm/blob/main/pyproject.toml).

## Optional extras

JAX FDM declares optional dependency groups you can install from a source checkout with `pip`:

```bash
pip install -e ".[viz]"    # 3D desktop viewer (compas_viewer), notebook viewer (compas_notebook) and 2D plotter (compas_plotter)
pip install -e ".[ipopt]"  # the IPOPT interior-point optimizer (cyipopt)
pip install -e ".[dev]"    # development tools (ruff, pytest, pre-commit, build, bump-my-version)
```

The `ipopt` extra might need a system Ipopt library available on your machine.

## Are you a Windows user?

JAX now provides official native CPU wheels for Windows, so JAX FDM should work directly.
On Windows you may also need to install the [Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

For GPU acceleration on Windows, native support is unavailable.
You can instead run JAX through the [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about), but keep in mind that it has no graphical output and that support for this configuration is experimental.
Please refer to [JAX's installation instructions](https://docs.jax.dev/en/latest/installation.html) for details.
