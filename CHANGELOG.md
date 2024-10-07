# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added support for kwargs in `LossPlotter.plot()`. The kwargs control the parameters of the equilibrium model used to plot the loss history.
- Added `VertexSupportParameter.index()`. This change might appear redundant, but it was necessary to deal with the method resolution order of the parent classes of `VertexSupportParameter`.
- Added `VertexGroupSupportParameter.index()` for similar reasons as the listed above.

### Changed

- Deprecated `jax_fdm.optimization.optimizers.scipy` in favor of `jax_fdm.optimization.optimizers.gradient_based`.
- Fixed bug in `EdgesForceEqualGoal.prediction()`: the normalization mean of compressive edge forces was a negative number. This led to negative normalized variance values, which was plainly incorrect. 
- `VertexGroupSupportParameter` inherits from `VertexGroupParameter` instead of `NodeGroupParameter`. This was a bug.

### Removed


## [0.8.5] 2024-09-15

### Added

### Changed

### Removed

- Removed `compas_singular` from dependencies list. That package has been archived and it is not used in `jax_fdm` source code; only in a couple examples.

## [0.8.4] 2024-05-09

### Added

- Added `NetworkSmoothGoal` to smoothen the shape of a network based on the fairness energy of its nodes w.r.t. their immediate neighborhood.
- Implemented `Graph.adjacency` to access the connectivity among nodes/vertices to compute new goals. 
- Added `adjacency_matrix` as numpy-only function to assemble `Graph.adjacency`. The function is largely inspired by `compas.matrices.adjacency_matrix`.

### Changed

- Now we use `time.perf_counter` instead of `time.time` to measure logic execution time more accurately.

### Removed


## [0.8.3] 2024-04-18

### Added

#### Goals
- Implemented `EdgeLoadPathGoal`.

### Changed

#### Floating-point arithmetic
- Updated how to enable double floating precision (`float64`) to comply with changes of `jax.config` in `jax==0.4.25`.

#### Equilibrium
- Fixed duplicated fixed point iteration in `EquilibriumModel.equilibrium_iterative`. This led to unnecessarily long runtimes. This change also fixes the "mysterious" bug that made `jaxopt` implicit differentiation incompatible with sparse matrices. 

#### Visualization
- `LossPlotter` exposes `plot_legend` to choose whether or not to show legend with curve labels.
- `FDNetworkArtist` takes absolute force density values to calculate viz colors in `"fd"` mode.
- Fixed bug in `FDNetworkViewerArtist` that threw error when inputing a custom list of edges to display. The problem was that the artist could not find the width of all the edges connected to a node because `edge_width` is only computed for the custom list of edges. The artist was expecting a dictionary with _all_ the edge widths.
- Package `compas-notebook` became an optional dependency because it does not support `compas<2.0` anymore.

### Removed


## [0.8.1] 2023-12-05

### Added

#### Losses
- Implemented `LogMaxError`. The goal of this error function is to work as a barrier soft constraint for target maximum values. One good use example would be to restrict the height of a shell to a maximum height. 

### Changed

### Removed


## [0.8.0] 2023-11-23

### Added

#### Models
- Added support for efficient reverse-mode AD of the calculation of equilibrium states in the presence of shape-dependent loads, via implicit differentiation. Forward-mode AD is pending.
- Added `EquilibriumModel.equilibrium_iterative` to compute equilibrium states that have shape-dependent edge and face loads using fixed point iteration.
- Added `EquiibriumModel.edges_load` and `EquiibriumModel.faces_load` to allow computation of edge and face loads.
- Implemented `EquilibriumModelSparse.stiffness_matrix`.
- Implemented `EquilibriumModel.stiffness_matrix`.
- Implemented `EquilibriumModel.force_matrix`.
- Implemented `EquilibriumModel.force_fixed_matrix`.
- Added `linearsolve_fn`, `itersolve_fn`, `implicit_diff`, and `verbose` as attributes of `EquilibriumModel`.
- Added `Equilibrium.load_nodes_iter` as attribute to include the node loads in `LoadState.nodes` when running `EquilibriumModel.equilibrium_iterative()`. Defaults to `False`.

#### Equilibrium
- Restored `vectors` field in `EquilibriumState`.
- Implemented `equilibrium.states.LoadState`.
- Implemented `equilibrium.states.EquilibriumParametersState`.

#### Solvers
- Implemented `solver_anderson`, to find fixed points of a function with `jaxopt.AndersonAcceleration`. The implicit differentiation operator of the solver provided by `jaxopt` is deactivated when using `EquilibriumModelSparse` because `jaxopt` does not support sparse matrices yet.
- Defined a `jax.custom_vjp` for `fixed_point`, an interface function that solves for fixed points of a function for different root-finding solver types: `solver_fixedpoint`, `solver_forward`, and `solver_newton`. 
- Implemented `solver_fixedpoint`, a function that wraps `jaxopt.FixedPointIterator` to calculate static equilibrium iteratively.
- Implemented `solver_forward`, to find fixed points of a function using an `equinox.while_loop`.
- Implemented `solver_netwon`, to find fixed points of a function using Newton's method.

#### Loads
- Added `equilibrium.loads` module to enable support for edge and face-loads, which correspond to line and area loads, respectively.
These two load types can be optionally become follower loads setting the `is_local` input flag to `True`. A follower load will update its direction iteratively, according to the local coordinate system of an edge or a face at an iteration. The two main functions that enable this feature are `loads.nodes_load_from_faces` and `loads.nodes_load_from_edges`. These functions are wrapped by `EquilibriumModel` under `EquiibriumModel.edges_load` and `EquiibriumModel.faces_load`. 
- Implemented `equilibrium.loads.nodes_`.

#### Datastructures
- Report standard deviation in `FDDatastructure.print_stats()`.
- Added constructor method `FDNetwork.from_mesh`.
- Added `FDMesh.face_lcs` to calculate the local coordinaty system of a mesh face.
- Added `datastructures.FDDatastructure.edges_loads`.
- Added `datastructures.FDMesh`.
- Added `datastructures.Datastructure`.
- Implemented `structures.EquilibriumStructureMeshSparse`.
- Implemented `structures.EquilibriumStructureMesh`.
- Implemented `structures.Mesh`.
- Implemented `structures.MeshSparse`.
- Implemented `structures.Graph`.
- Implemented `structures.GraphSparse`.
- Added `FDNetwork.is_edge_fully_supported`. 
- Added `EquilibriumMeshStructure.from_mesh` with support for inhomogenous faces (i.e. faces with different number of vertices). The solution is to pad the rows of the `faces` 2D array with `-1` to match `max_num_vertices`. 

#### Goals

- Implemented `NetworkXYZLaplacianGoal`.
- Implemented base class `MeshGoal`.
- Implemented `MeshXYZLaplacianGoal`.
- Implemented `MeshXYZFaceLaplacianGoal`.
- Implemented `MeshAreaGoal`.
- Implemented `MeshFacesAreaEqualizeGoal`.

#### Optimization
- Added `optimization.Optimizer.loads_static` attribute to store edge and face loads during optimization.

#### Geometry
- Added `polygon_lcs` to compute the local coordinate system of a closed polygon.
- Added `line_lcs` to compute the local coordinate system of a line.
- Added `nan` gradient guardrail to `normalize_vector` calculations.

#### Parameters
- Added support for mesh vertex parameters.

#### Numerical
- Added explicit array integer types in `__init__`: `DTYPE_INT_NP` and `DTYPE_INT_JAX`

#### Sparse solver
- Set `spsolve_gpu_ravel` as the default sparse solver on GPUs (`spsolve_gpu`).
- Added `spsolve_gpu_ravel` to solve the FDM linear system all at once on GPUs.
- Implemented helper function `sparse_blockdiag_matrix` to `spsolve_gpu_ravel`.

#### Visualization
- Added `plotters/VectorArtist` to custom plot loads and reactions arrows.
- Implemented `LossPlotter._print_error_stats` to report loss breakdown of error terms.

### Changed

#### Models

#### Equilibrium
- The functions `fdm` and `constrained_fdm` take iterative equilibrium parameters as function arguments.
- The functions `fdm` and `constrained_fdm` can take an `FDMesh` as input, in addition to `FDNetwork`.

#### Sparse solver
- Decoupled `sparse_solver` from any force density calculations. Now, it is a simpler solver that only takes as inputs the LHS matrix `A` and the RHS matrix `b`, and thus, it could be used to potentially solve any sparse linear system of equations. Its signature now is analogous to that of `jax.numpy.linalg.solve`.
- Condensed signature of sparse linear solver `sparse_solve` to take a structure `EquilibriumStructure` as input, instead of explicit attributes of a structure.
- Changed signature of `sparse_solve_bwd` to take two arguments, where the first is the "residual" values produced on the forward pass by ``fwd``, and the second is the output cotangent with the same structure as the primal function output (`sparse_solve`).
- Condensed signature of helper functions `sparse_solve_fwd` to take matrices `A` and `b` as inputs, instead of explicit attributes of the FDM and of a `EquilibriumStructure`.
- Renamed previous verison of `spsolve_gpu` to `spsolve_gpu_stack`.

#### Geometry
- Added support for `jnp.nan` inputs in the calculations of `geometry.normal_polygon`. 

#### Losses
- Changed signature of `Regularizer.__call__` to take in parameters instead of equilibirum state. 

#### Datastructures 
- Overhauled `EquilibriumStructure` and `EquilibriumStructureSparse`. They are subclasses `equinox.Module`, and now they are meant to be immutable. They also have little idea of what an `FDNetwork` is.
- Modified `face_matrix` adjacency matrix creation function to skip -1 vertices. This is to add support for `MeshStructures` that have faces with different number of vertices.

#### Optimization
- Use `jax.value_and_grad(loss_fn(x))` instead of using `loss_fn(x)` and `jax.grad(loss_fn(x))` separately. This results in optimization speedup because we get both value and grad with a single VJP call. 
- `Optimizer.problem` takes an `FDNetwork` as input.
- `Optimizer.problem` takes boolean `jit_fn` as arg to disable jitting if needed.
- Changed `ParameterManager` to require an `FDNetwork` as argument at initialization.
- Changed `Parameter.value` signature. Gets value from `network` directly, not from `structure.network`
- `optimization.OptimizationRecorder` has support to store, export and import named tuple parameters.

#### Visualization
- Fixed bug in `viewers/network_artist.py` that overshifted load arrows.
- Edge coloring considers force sign for `force` color scheme in `artists/network_artist.py`.
- Fixed bug with the coloring of reaction forces in `viewers/network_artist.py`.
- Fixed bug with the coloring of reaction forces in `artists/network_artist.py`.
- `LossPlotter` has support to plot named tuple parameters.


### Removed

- Removed `EquilibriumModel.from_network`.
- Removed `sparse.force_densities_to_A`. Superseded by `EquilibriumModelSparse.stiffness_matrix`. 
- Removed `sparse.force_densities_to_b`. Superseded by `EquilibriumModel.force_matrix`.
- Removed partial jitting from `Loss.__call__`.
- Removed partial jitting from `Error.__call__`.


## [0.7.1] 2023-05-08

### Added

### Changed

- Fixed signature bug in constraint initialization in `ConstrainedOptimizer.constraints`.

### Removed

- Removed implicit `partial(jit)` decorator on `ConstrainedOptimizer.constraint`. Jitting now takes place explicitly in `ConstrainedOptimizer.constraints`.


## [0.7.0] 2023-05-08

### Added
 
- Added `EquilibriumStructure.init` as a quick fix to warm start properties.

### Changed

- Changed signature of equilibrium model to be `EquilibriumModel(params, structure)`.
- The `init` function in goals, constraints and parameter takes `(model, structure)` as arguments.
- Removed `connectivity` related operations from `EquilibriumModel` and inserted them into `EquilibriumStructure`.
- Fixed bug in `EquilibriumStructure.nodes` that led to a recursive timeout error.
- Renamed example file `butt.py` to `vault.py`/
- Renamed file `optimizers.py` to `scipy.py` in `jax_fdm.optimization`.

### Removed

- Removed `structure` attribute from `EquilibriumModel` en route to equinox modules. 

## [0.6.0] 2023-04-30

### Added

Support for differentiable CPU sparse solver
- Added support for differentiable CPU and GPU sparse solvers to compute the XYZ coordinates of the free nodes with FDM. The solver is custom made and registered via a `jax.custom_vjp`. The forward and backward passes of the sparse solver were made possible thanks to the contributions of @denizokt. This solver lives in `equilibrium.sparse.py`.
- Added `spsolve_gpu` and `spsolve_cpu`. The sparse solver backend is different (the former uses CUDA and the latter scipy) and it gets selected automatically based on what the value of `jax.default_device` is at runtime with function `register_sparse_solver`.
- Implemented `EquilibriumStructureSparse`.
- Division of responsabilities: created a `EquilibriumModelSparse` that solves for static equilibrium using an `EquilibriumStructureSparse` and the sparse solver. `EquilibriumModel` implements the dense linear solver.

### Changed

- `LossPlotter` instantiates a dense `EquilibriumModel`.

### Removed


## [0.5.2] 2023-03-15

### Added

### Changed

### Removed

- Removed `DTYPE` from `combine_parameters` signature in `jax_fdm.parameters` due to clashes with JAX versions on windows.

## [0.5.1] 2023-03-10

### Added

### Changed

- Added `ParameterManager.indices_opt_sort` to fix bug that mistmatch optimization values for unconsecutive parameter indices. This bug applied to individual parameters and to parameter groups.

### Removed


## [0.5.0] 2023-03-02

### Added

- Implemented parameter groups!
- Added `EdgesForceDensityParameter`.
- Added `NodesLoadXParameter`, `NodesLoadYParameter`, `NodesLoadZParameter`.
- Added `NodesSupportXParameter`, `NodesSupportYParameter`, `NodesSupportZParameter`.
- Added `EdgesForceEqualGoal` to goals.
- Implemented `area_polygon` in `jax_fdm.geometry`.
- Added `FDNetwork.number_of_supports()` to count number of supported nodes in a network.
- Added `network_validate` to check the validity of a network before equilibrium calculations.

### Changed

- Sped up `EquilibriumModel.nodes_free_positions()` computation after replacing `jnp.diag(q)` with `vmap(jnp.dot)(q, *)`. 
- Vectorized error computations in `LossPlotter.plot()` to expedite method.
- `OptimizationRecorder.record()` now stores history in a dictionary, not in a list. 
- Fixed bug in `FDNetworkViewerArtist` that ocurred while plotting reaction forces on unconnected nodes.
- Turned `TrustRegionConstrained` into a first order optimizer.

### Removed

- Removed `LossPlotter.network` attribute.
- Replaced `NodeAnchorXParameter` with `NodeSupportXParameter`.
- Replaced `NodeAnchorYParameter` with `NodeSupportYParameter`.
- Replaced `NodeAnchorZParameter` with `NodeSupportZParameter`.

## [0.4.5] 2023-02-04

### Added

- Added `Goal.is_collectible` flag to determine if a goal should form part of an optimization `Collection`.
- Implemented `EdgesLengthEqualGoal`.
- Added `AbsoluteError` term.
- Added `MeanAbsoluteError` terms.

### Changed

- Fixed omission of `gstate.weight` when computing `PredictionError`.

### Removed


## [0.4.4] 2022-12-15

### Added

- Implemented `LBFGSBS` to ensure compatibility of JAX gradients with scipy.

### Changed

### Removed


## [0.4.3] 2022-12-14

### Added

- Implemented `NotebookViewer` to visualize networks in jupyter notebooks.
- Added `FDNetworkNotebookArtist` to visualize networks in jupyter notebooks.
- Added `noteboks.shapes.Arrow` to display force vectors as meshed arrows.

### Changed

### Removed


## [0.4.2] 2022-12-12

### Added

- Implemented `NodeTangentAngleGoal`.
- Implemented `NodeNormalAngleGoal`.
- Implemented `NodeTangentAngleConstraint`.
- Implemented `NodeNormalAngleConstraint`.
- Added `EquilibriumStructure.connectivity_faces` to optimize for face-related quantities in an `FDNetwork`.

### Changed

- Fixed bug in `edgewidth` sizing in `NetworkArtist` when all edges have the same force. 
- Changed `angle_vectors` to return angle in radians by default.
- Shifted loads start point in `plotters.NetworkArtist` for them to be incident to the nodes.

### Removed


## [0.4.1] 2022-11-29

### Added

### Changed

- Changed generator unpacking `*sarrays` in `parameters.split` unsupported in Python 3.7. 
- Changed tension-compression force color map gradient to a binary color map.

### Removed


## [0.4.0] 2022-11-22

### Added

- Added `goals.NodeZCoordinateGoal`.
- Added `goals.NodeYCoordinateGoal`.
- Added `goals.NodeXCoordinateGoal`.
- Added `constraints.NodeZCoordinateConstraint`.
- Added `constraints.NodeYCoordinateConstraint`.
- Added `constraints.NodeXCoordinateConstraint`.
- Added `IPOPT`, a second-order, constrained optimizer that wraps `cyipopt`, to the repertoire of optimizers.

### Changed

- Restructured `optimization.optimizers` into separate files.
- Renamed `goals.edgegoal` submodule to `goals.edge`.
- Renamed `goals.nodegoal` submodule to `goals.node`.
- Renamed `goals.networkgoal` submodule to `goals.network`.
- Enabled support for one-sided bounds in `constraint.Constraints`.
- `NetworkLoadPathGoal` uses `jnp.multiply` instead of multipication operator `*`.
- Broke down `Optimizer.minimize()` into `.problem()` and `.solve()`.
- For efficiency, `SecondOrderOptimizer` calculates `hessian` as `jax.jacfwd(jax.jacrev)`.
- Changed calculation of the scalar ouput of `Loss` using a compact loop. 
- Changed calculation of the scalar ouput of `Error` using a compact loop. 

### Removed


## [0.3.0] 2022-11-08

### Added
- Implemented `jax_fdm.parameters` to choose optimization parameters a la carte!
- Created `parameters.EdgeForceDensityParameter`.
- Created `parameters.NodeAnchorXParameter`.
- Created `parameters.NodeAnchorYParameter`.
- Created `parameters.NodeAnchorZParameter`.
- Created `parameters.NodeLoadXParameter`.
- Created `parameters.NodeLoadYParameter`.
- Created `parameters.NodeLoadZParameter`.
- Implemented `EquilibriumStructure.nodes`.
- Implemented `EquilibriumStructure.edges`.
- Added `EquilibriumStructure.anchor_index`.
- Implemented `ParameterManager.parameters_ordered` to fix order mismatch between optimization parameters and their bounds.

### Changed

- `EquilibriumModel.__call__` tasks `q`, `xyz_fixed` and `loads` as arguments.
- `FDNetwork.transform` only modifies node coordinates.
- `FDNetworkViewerArtist` shifts a node load by the maximum edge width at the node.
- `OptimizationRecorder.history` stores `q`, `xyz_fixed` and `loads`. 
- `LossPlotter.plot` supports `q`, `xyz_fixed` and `loads` to be compatible with `OptimizationRecorder.history`.

### Removed


## [0.2.4] 2022-10-27

### Added

- Enabled the creation of animations with mesh representations of the elements of a `FDNetwork`!
- `FDNetworkViewerArtist` implements `update_*()` methods to update objects in the `Viewer`.
- `FDNetworViewerArtist` implements `add_*()` methods to add drawn objects to `Viewer`.
- `FDNetworkArtist` now stores network element collections as attributes.

### Changed


### Removed

- Removed `FDNetworkPlotterArtist.draw_loads()` as it is handled by parent artist class.
- Removed `FDNetworkPlotterArtist.draw_reactions()` as it is handled by parent artist class.

## [0.2.3] 2022-10-25

### Added

- Implemented `SecondOrderOptimizer`, which computes the hessian of a loss function.
- Added support for the `NetwtonCG` and `LBFGSB` scipy optimizers.
- Created `NodeSegmentGoal`
- Implemented `closest_point_on_segment` using `lax.cond`

### Changed

- Pinned viewer dependency to `compas_view2==0.7.0`

### Removed

- Removed `Viewer.__init__()`

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

