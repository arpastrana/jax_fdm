<h1 align='center'>JAX FDM</h1>

<!-- Badges -->
![build](https://github.com/arpastrana/jax_fdm/workflows/build/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7258292.svg)](https://doi.org/10.5281/zenodo.7258292)
[![PyPI - Latest Release](https://img.shields.io/pypi/v/jax-fdm.svg)](https://pypi.python.org/project/jax-fdm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jax-fdm.svg)](https://pypi.python.org/project/jax-fdm)
[![arXiv](https://img.shields.io/badge/arXiv-2307.12407-b31b1b.svg)](https://arxiv.org/abs/2307.12407)
[![CMAME](https://img.shields.io/badge/CMAME-10.1016%2Fj.cma.2026.118783-blue.svg)](https://doi.org/10.1016/j.cma.2026.118783)
<!-- [![GitHub - License](https://img.shields.io/github/license/arpastrana/jax_fdm.svg)](https://github.com/arpastrana/jax_fdm) -->

A differentiable, hardware-accelerated framework for the structural design of lightweight structures.

> Crafted with care in the [AI Lab](http://ai.princeton.edu/) at [Princeton University](https://princeton.edu) ❤️🇺🇸

![](images/jax_logo.gif)

Lightweight span long distances with slender cross-sections due to their mechanically efficient shapes.
However, simulating these structures and turning them into feasible designs that satisfy technical constraints remains challenging due to geometrically nonlinear mechanical behaviors and high-dimensional search spaces.

JAX FDM enables the solution of inverse problems for lightweight structures modeled as pin-jointed bar systems using the force density method (FDM) and gradient-based optimization.
It streamlines the integration of mechanical simulations into deep learning models for machine learning research.

## Key features

- **Legendary form-finding solver.**
JAX FDM computes static equilibrium states for pin-jointed bar systems with the [force density method (FDM)](https://www.sciencedirect.com/science/article/pii/0045782574900450), the time-tested form-finding solver for geometrically nonlinear systems backed up by over 50 years of peer-reviewed research 📚.
<!--  -->
- **Derivatives, JIT compilation and parallelization.**
JAX FDM is written in [JAX](https://github.com/google/jax), a library for high-performance numerical computing and machine learning research, and it thus inherits many of JAX's perks: calculate derivatives, parallelize, and just-in-time (JIT) compile entire form-finding simulations written in Python code, and run them on a CPU, a GPU, or a TPU 🤯.
- **Autotune those force densities, loads and supports.**
A form-found structure should fulfill additional design requirements to become a feasible structure.
Formulate an inverse form-finding scenario like this as an optimization problem with JAX FDM.
Then, let one of its gradient-based optimizers solve the problem by automatically tweaking the system's force densities, applied loads and support positions 🕺🏻.
- **A rich bank of goals, constraints and loss functions.**
No two structures are alike.
JAX FDM allows you to model a custom inverse form-finding problem with its (growing!) collection of goals, constraints, and loss functions via a simple, object-oriented API.
The available goals and constraints in the framework are granular and applicable to an entire structure; to a subset of its nodes (i.e., vertices), edges, and combinations thereof 💡.
<!-- Don't see a goal or a constraint you fit?. Add yours with ease! Consult our documentation guide (in progress) to see how you add yours. -->
- **Form-finding simulations as another layer in a neural network.**
As an auto-differentiable library, JAX FDM can be seamlessly added as a layer in a differentiable function approximator like a neural network that can be then trained end-to-end.
Let the neural network learn the underlying physics of static equilibrium *directly* from the form-finding simulation, instead of resorting to laborious techniques like data augmentation 🤖.

JAX FDM is a research project under development.
Expect sharp edges and possibly some API breaking changes as we continue to support a broader set of features.

## Installation

First, create a new [Anaconda](https://www.anaconda.com/) environment and then activate it:

```bash
conda create -n jaxenv
conda activate jaxenv
```

Next, install COMPAS and COMPAS VIEW2 via `conda`:

```bash
conda install -c conda-forge compas==1.17.10 compas_view2==0.7.0 
```

Finally, install JAX FDM with a one-liner via `pip`:

```bash
pip install jax-fdm
```

JAX FDM requires Python 3.10+ and builds on JAX, NumPy, SciPy, Equinox, and the COMPAS framework. See `requirements.txt` for the complete dependency list.
For visualization, it uses COMPAS_VIEW2 0.7.0.

#### Optional extras

JAX FDM declares optional dependency groups you can install from a source checkout with `pip`:

```bash
pip install -e ".[viz]"    # 3D viewer (compas_view2) and matplotlib
pip install -e ".[ipopt]"  # the IPOPT interior-point optimizer (cyipopt)
pip install -e ".[dev]"     # development tools (ruff, pytest, build, bump-my-version)
```

Note that `compas_view2` is distributed through `conda-forge`, so the `viz` extra may still require a `conda` install as shown above. The `ipopt` extra needs a system Ipopt library available on your machine.

### Are you a Windows user? 
 
JAX now provides official native CPU wheels for Windows, so JAX FDM should work directly.
On Windows you may also need to install the [Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

For GPU acceleration on Windows, native support is unavailable.
You can instead run JAX through the [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about), but keep in mind that it has no graphical output and that support for this configuration is experimental.
Please refer to [JAX's installation instructions](https://docs.jax.dev/en/latest/installation.html) for details.

## Quick example

Suppose you are interested in generating a form in static equilibrium for a 10-meter span arch subjected to vertical point loads of 0.3 kN.
The arch has to be a compression-only structure.
You model the arch as a `jax_fdm` network (download the arch `json` file [here](https://github.com/arpastrana/jax_fdm/blob/main/data/json/arch.json)).
Then, you apply a force density of -1 to all of its edges, and compute the required shape with the force density method. 

```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm


network = FDNetwork.from_json("data/json/arch.json")
network.edges_forcedensities(q=-1.0)
network.nodes_supports(keys=[node for node in network.nodes() if network.is_leaf(node)])
network.nodes_loads([0.0, 0.0, -0.3])

f_network = fdm(network)
```

You now wish to find a new form for this arch that minimizes the [total Michell's load path](https://doi.org/10.1007/s00158-019-02214-w), while keeping the length of the arch segments between 0.75 and 1 meters.
You solve this constrained form-finding problem with the SLSQP gradient-based optimizer.

```python
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import PredictionError
from jax_fdm.losses import Loss


loss = Loss(PredictionError(goals=[NetworkLoadPathGoal()]))
constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]
optimizer = SLSQP()

c_network = constrained_fdm(network, optimizer, loss, constraints=constraints)
```

You finally visualize the unconstrained form-found arch `f_network` (gray) and the constrained one, `c_network` (in teal) with the `Viewer`.

```python
from jax_fdm.visualization import Viewer


viewer = Viewer(width=1600, height=900)
viewer.add(c_network)
viewer.add(f_network, as_wireframe=True)
viewer.show()
```

![](images/arch_loadpath.png)

The constrained form is shallower than the unconstrained one as a result of the optimization process.
The length of the arch segments also varies within the prescribed bounds to minimize the load path: segments are the longest where the arch's internal forces are lower (1.0 meter, at the apex); and conversely, the segments are shorter where the arch's internal forces are higher (0.75 m, at the base).

## Documentation

Documentation is a work in progress. In the meantime, check out the scripts in the [`examples/`](https://github.com/arpastrana/jax_fdm/tree/main/examples) folder.

## More examples


### Notebooks

> These notebooks run directly from your browser without having to install anything locally! 

- [Arch](https://colab.research.google.com/drive/1_SrFuRPWxB0cG-BaZtNqitisQ7M3oUOG?usp=sharing): Control the height and the horizontal projection of a 2D arch.
- [3D spiral](https://colab.research.google.com/drive/13hi9VsQ2PSLY2otfyDSvlX3xhpfFJ7zJ?usp=sharing): Calculate the loads required to maintain a compression-only 3D spiral in equilibrium [(Angelillo, et al. 2021)](https://doi.org/10.1016/j.engstruct.2021.112176).
- [Creased masonry vault](https://colab.research.google.com/drive/1I3ntFbAqmxDzLmTwiL8z-pYoiZLC1x-z?usp=sharing): Best-fit a target surface [(Panozzo, et al. 2013)](https://cims.nyu.edu/gcl/papers/designing-unreinforced-masonry-models-siggraph-2013-panozzo-et-al.pdf).


### Scripts

> These python scripts require a local installation of JAX FDM.

- [Pointy dome](https://github.com/arpastrana/jax_fdm/blob/main/examples/dome/dome.py): Control the tilt and the coarse width of a brick dome. 
- [Triple-branching saddle](https://github.com/arpastrana/jax_fdm/blob/main/examples/monkey_saddle/monkey_saddle.py): Design the distribution of thrusts at the supports of a monkey saddle network while constraining the edge lengths.
- [Saddle bridge](https://github.com/arpastrana/jax_fdm/blob/main/examples/pringle/pringle.py): Create a crease in the middle of the bridge while constraining to transversal edges of the network to a target plane. 

## Citation

If you found this library to be useful in academic or industry work, please consider 1) starring the project on Github, and 2) citing it:


``` bibtex
@article{pastrana_dfdm_2026,
         title = {Differentiable force density method for the design of lightweight structures},
         author = {Pastrana, Rafael and Oktay, Deniz and Bletzinger, Kai-Uwe and Adams, Ryan P. and Adriaenssens, Sigrid},
         date = {2026},
         journaltitle = {Computer Methods in Applied Mechanics and Engineering},
         volume = {458},
         pages = {118783},
         issn = {00457825},
         doi = {10.1016/j.cma.2026.118783}}
```

``` bibtex
@inproceedings{pastrana_jaxfdm_2023,
               title = {{{JAX FDM}}: {{A}} differentiable solver for inverse form-Finding},
               booktitle = {Differentiable {{Almost Everything Workshop}} of the 40th {{International Conference}} on {{Machine Learning}}},
               author = {Pastrana, Rafael and Oktay, Deniz and Adams, Ryan P. and Adriaenssens, Sigrid},
               year = {2023},
               address = {Hawaii, USA},
               url = {https://openreview.net/forum?id=Uu9OPgh24d}}
```

```bibtex
@software{pastrana_jaxfdm_software_2023,
          title={{JAX~FDM}: {A}uto-differentiable and hardware-accelerated force density method},
          author={Rafael Pastrana and Deniz Oktay and Ryan P. Adams and Sigrid Adriaenssens},
          year={2023},
          doi={10.5281/zenodo.7258292},
          url={https://github.com/arpastrana/jax\_fdm}}
```

## Acknowledgements

This work has been supported by the **U.S. National Science Foundation** under grant **OAC-2118201** and the [Institute for Data Driven Dynamical Design](https://www.mines.edu/id4/).

## See also

[COMPAS CEM](https://github.com/arpastrana/compas_cem): Inverse design of 3D trusses with the combinatorial equilibrium modeling (CEM) framework.

[JAX CEM](https://github.com/arpastrana/jax_cem): The combinatorial equilibrium modeling (CEM) framework in JAX.

[JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs.

## License

MIT
