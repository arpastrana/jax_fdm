# Installation

JAX FDM supports Python 3.11 to 3.13 on Linux, macOS and Windows.
It builds on JAX, SciPy, Equinox, and the COMPAS framework.
See the complete [dependency list](https://github.com/arpastrana/jax_fdm/blob/main/pyproject.toml).

## Install

Install JAX FDM from PyPI with a one-liner via `pip`:

```bash
pip install jax-fdm
```

This pulls in COMPAS 2.x and the other core dependencies automatically.

To also get the visualization tools, the 3D desktop viewer (`compas_viewer`), the notebook viewer (`compas_notebook`) and the 2D plotter (`compas_plotter`), ask for the `viz` extra:

```bash
pip install "jax-fdm[viz]"
```

If you manage your project with [uv](https://docs.astral.sh/uv/), the equivalent is:

```bash
uv add "jax-fdm[viz]"
```

## Develop

To work on JAX FDM itself, clone the repository and let `uv` set up the environment.
It installs a matching Python, the package in editable mode, the visualization extra and every development tool, pinned by the committed `uv.lock`, in one step:

```bash
git clone https://github.com/arpastrana/jax_fdm.git
cd jax_fdm
uv sync --all-groups --extra viz
uv run invoke test
```

Prefix commands with `uv run` and they execute inside the project environment, with no activation step.
A bare `uv sync` installs only the runtime dependencies, the same as `pip install -e .`.
Contributor tooling lives in `[dependency-groups]` and is never published to PyPI, so a plain `pip install jax-fdm` never drags it in.

If you prefer `pip`, create and activate a virtual environment on Python 3.11 to 3.13 and install the same groups by name.
This needs `pip` 25.1 or newer:

```bash
pip install -e ".[viz]" --group dev --group docs --group typecheck
```

See the [contributing guide](https://github.com/arpastrana/jax_fdm/blob/main/CONTRIBUTING.md) for the rest of the workflow.

## Are you a Windows user?

JAX now provides official native CPU wheels for Windows, so JAX FDM should work directly.
On Windows you may also need to install the [Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

For GPU acceleration on Windows, native support is unavailable.
You can instead run JAX through the [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about), but keep in mind that it has no graphical output and that support for this configuration is experimental.
Please refer to [JAX's installation instructions](https://docs.jax.dev/en/latest/installation.html) for details.
