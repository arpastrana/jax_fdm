# More examples

## Jupyter notebooks

These notebooks run directly from your browser without having to install anything locally! Their sources live in the [`notebooks/`](https://github.com/arpastrana/jax_fdm/tree/main/notebooks) folder.

- [Arch](https://colab.research.google.com/github/arpastrana/jax_fdm/blob/main/notebooks/arch.ipynb): Control the height and the horizontal projection of a 2D arch.
- [3D spiral](https://colab.research.google.com/github/arpastrana/jax_fdm/blob/main/notebooks/spiral.ipynb): Calculate the loads required to maintain a compression-only 3D spiral in equilibrium [(Angelillo, et al. 2021)](https://doi.org/10.1016/j.engstruct.2021.112176).
- [Creased masonry vault](https://colab.research.google.com/github/arpastrana/jax_fdm/blob/main/notebooks/vault.ipynb): Best-fit a target surface [(Panozzo, et al. 2013)](https://cims.nyu.edu/gcl/papers/designing-unreinforced-masonry-models-siggraph-2013-panozzo-et-al.pdf).

## Python scripts

The scripts require a local installation of JAX FDM.

- [Pointy dome](https://github.com/arpastrana/jax_fdm/blob/main/examples/dome/dome.py): Control the tilt and the coarse width of a brick dome.
- [Triple-branching saddle](https://github.com/arpastrana/jax_fdm/blob/main/examples/monkey_saddle/monkey_saddle.py): Design the distribution of thrusts at the supports of a monkey saddle network while constraining the edge lengths.
- [Saddle bridge](https://github.com/arpastrana/jax_fdm/blob/main/examples/pringle/pringle.py): Create a crease in the middle of the bridge while constraining the transversal edges of the network to a target plane.
