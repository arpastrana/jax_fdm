# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

### Changed

### Removed


## [0.2.2] 2022-10-17

### Added

- Added `datastructures.FDNetwork.transformed()` 
- Created `visualization.plotters.FDNetworkPlotterArtist`
- Implemented `visualization.plotters.Plotter`

### Changed

### Removed


## [0.2.1] 2022-10-17

### Added

### Changed

- Rolled back support for python `3.7`.

### Removed


## [0.2.0] 2022-10-11

### Added

- Implemented `visualization.viewers.Viewer` as a thin wrapper around `compas_view2.App`.
- Created `visualization.plotters.LossPlotter`.
- Implemented `visualization.viewers.FDNetworkViewerArtist`.
- Implemented `visualization.artists.FDNetworkArtist`.
- Created `jax_fdm.visualization` module.
- Implemented `FDNetwork.print_stats()`.
- Implemented `FDNetwork.node_reaction()` to get the reaction force at a fixed node.
- Added `FDNetwork.nodes_fixed()` to query nodes with a support.

### Changed

### Removed


## [0.1.2] 2022-09-30

### Added

- Created beta release! 🎉 Since we were busy adding new features to JAX FDM, we forgot to log them. Sorry! We promise to do better from here on. Please check the git log (fairly granular) for details on the all the features.

### Changed

### Removed

