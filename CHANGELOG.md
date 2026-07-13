# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Pointed the docs' Colab notebook links at the repo instead of Google Drive copies: the "Jupyter notebooks" list in `docs/examples.md` now uses `colab.research.google.com/github/...` URLs that open `notebooks/{arch,spiral,vault}.ipynb` straight from the repo, so Colab always loads the current committed (COMPAS 2.x) version and the stale Drive uploads are retired. Each notebook's install cell is now active (`!pip install -q -U jax-fdm compas_notebook`) so "Open in Colab → Run all" works on a fresh runtime.
- Deduplicated the README against the docs site: the deeper content (optional extras, Windows notes, the full quick example, the notebook/script galleries) now lives only in the docs, and the README links out to it. The pitch and key features are wrapped in `pymdownx.snippets` section markers — invisible on GitHub — and transcluded into `docs/index.md` at build time, making the README the single source of that prose.

### Added

- Added logo as svg for navbar and favicon.
- Split the visualization API page into one page per backend — 3D viewer, notebook viewer, and 2D plotter (which also carries the loss plotter) — nested under a foldable "Visualization" group in the docs navigation. The landing page for the group stays as an overview that routes to the three backends.
- The docs citation page now generates its BibTeX from `CITATION.cff` at build time, via an mkdocs hook (`docs/hooks/citation.py`) that replaces a marker comment in `docs/citation.md` with `@article`/`@inproceedings`/`@software` entries templated from the cff — so the citation metadata has one source of truth and the docs page cannot drift from it. The README's hand-written BibTeX must stay (GitHub renders it raw), so a new local pre-commit hook runs the same file as a script to fail any commit where the README stops citing the cff's preferred DOI. `mkdocs serve` watches `CITATION.cff` for live reload.
- Added a `CITATION.cff` file, which turns on GitHub's "Cite this repository" sidebar button. Its `preferred-citation` points at the CMAME article (Pastrana et al. 2026, doi:10.1016/j.cma.2026.118783), so the button hands out the paper's citation rather than a generic software one; the software record itself carries the Zenodo DOI. The hand-written BibTeX blocks in the README and the docs citation page remain the copy-paste artifacts.
- Revived the dead `docs.yml` workflow (its jobs block had been commented out for years): pull requests to main run `mkdocs build --strict` as a check, and pushes deploy versioned docs to GitHub Pages with [mike](https://github.com/jimporter/mike), replacing the stale 2023 Sphinx build on `gh-pages`. Every push to main updates a rolling `dev` version; every release tag `vX.Y.Z` publishes `X.Y.Z` and repoints the `latest` alias, which is where the site's root URL lands. The Material theme shows a version dropdown in the header, fed by mike's `versions.json`. The runners install `.[viz,docs]` plus the system GL libraries PySide6 needs headless, and set `JAX_PLATFORMS=cpu`. A docs badge linking to the site joined the README, the broken `github.com//jax_fdm` links in CONTRIBUTING.md are repaired, and `invoke docs` exists again (`mkdocs build --strict`, or `mkdocs serve` via `--serve`) — CONTRIBUTING had kept advertising the task after its Sphinx-era implementation was deleted.
- Added an API reference to the docs site: an overview page that walks the subpackages in workflow order, plus one page per subpackage (`docs/api/`). The pages are hand-written mkdocstrings stubs grouped thematically (e.g. goals by element type, optimizers by family), so each documented object is listed explicitly by its canonical module path — the dynamic star-import `__all__`s are never slurped wholesale, which keeps re-exported third-party names and submodule aliases out of the reference. The `custom_vjp`-wrapped implicit solvers and the equinox `Module` structures render correctly under griffe's static analysis.
- Replaced the dead Sphinx skeleton in `docs/` with an mkdocs + mkdocs-material + mkdocstrings site (`strict` builds, violet-branded Material theme, MathJax, per-page permalinks). The landing, installation, citation and license pages are distilled from the README; the license page embeds `LICENSE` via `pymdownx.snippets`. A new `docs` optional-dependency group pins the mkdocs/mkdocstrings trio exactly, and `[project.urls]` grew a `Documentation` entry pointing at the future GitHub Pages URL. The API reference stubs and the docs CI workflow land separately.
- Ported the 2D plotter backend to the revived, standalone `compas_plotter` (COMPAS 2.x, scene-object based). `FDNetworkPlotterObject` and `FDMeshPlotterObject` draw a force density datastructure on the plotter's matplotlib canvas with the shared FD display semantics of `visualization/style.py`: edge color modes, per-edge force-remapped widths (default bounds `(0.5, 5.0)` points), support coloring, and load and reaction arrows as data-space polygons proportional to the arrow length, batched into one collection per category and reported to `plotter.zoom_extents()`. The datastructure vocabularies carry over from the other backends; free nodes/vertices default to white on the plotter (the 3D backends keep grey), and the arrows keep one marker radius of clearance from the points they act on. `compas_plotter` is part of the `viz` extra; without it `Plotter` degrades to the same warn-and-absorb null object as the other optional backends.
- Rebuilt the visualization module on native COMPAS 2.x scene objects. `FDNetworkObject`/`FDMeshObject` (compas_viewer) and `ThreeFDNetworkObject`/`ThreeFDMeshObject` (compas_notebook) render a force density datastructure — its force density metadata (edge colors and widths, support colors, load and reaction vectors) plus a shaded surface for mesh faces — as one parent scene object with one batched child per element category (Edges, Nodes/Vertices, Loads, Reactions, and Faces for meshes), so visibility toggles and opacities work per category. `viewer.add(...)` returns the scene object; an animation loop calls `obj.update()` on it. The FD display semantics (edge color modes, force-remapped widths, load/reaction arrow placement) live in `visualization/style.py` as backend-neutral functions shared by both backends.
- Converted the notebook backend to compas_notebook 2.x (`>=0.11`). `NotebookViewer` now subclasses `compas_notebook.viewer.Viewer` and folds the `width`/`height`/`show_grid`/`viewport`/`camera_position`/`camera_target` kwargs into a `compas_notebook.config.Config`. The notebook scene objects batch each element category into a single pythreejs buffer with per-element colors to keep the Jupyter widget count low; rendering is draw-once (no scene tree, no animation loop).
- Registered `FDNetwork` and `FDMesh` with the COMPAS 2.x scene system, so a native COMPAS scene — a bare `compas_viewer.Viewer` or a compas_notebook scene — renders them through the jax_fdm scene objects.
- Aliased `edgewidth` to `linewidth` for plain-geometry adds in the 3D `Viewer`, so `edgewidth` is the one edge-width vocabulary across FD and plain adds alike. An explicit `linewidth=` still wins.
- Committed the sources of the three Colab notebooks linked in the README to a new `notebooks/` folder, and migrated them to the COMPAS 2.x API.
- The 3D viewer renders every element of a force density datastructure (edges, nodes/vertices, load and reaction arrows) as its own scene object by default, grouped per category in the scene tree, so single elements are clickable, highlightable and foldable. Fused category batching remains available as an explicit opt-in via `viewer.add(obj, fuse=True)` for fast loading and animation; both render paths build their triangle soups through the same buffer builders, so they are vertex-identical.
- Selecting a force density element in the 3D viewer shows its attributes in the sidebar's Object tab, read live from the datastructure — an edge shows key, force density, force and length; a node/vertex shows key, coordinates, support flag, load and reaction; an arrow shows its anchor point and vector — plus a one-line summary in the status bar. The readout is read-only.
- The viewer hints at `fuse=True` when a per-element add crosses 1000 elements, and warns once when `@viewer.on` starts an animation loop with per-element objects in the scene (their buffers update one by one per frame).

### Changed

- **Breaking:** `VertexNormalAngleGoal` and `VertexTangentAngleGoal` predict signed angles again, reverting the 0.11.0 fold into `[0, pi/2]`. The normal angle comes from the signed cosine and spans `[0, pi]`, so it is covariant with the orientation of the averaged vertex normal: a normal within 90 degrees of the reference vector reads acute, one folded past it reads obtuse. The tangent angle (`pi/2 - angle_normal`) accordingly spans `[-pi/2, pi/2]` and its sign tells a surface rising toward the reference vector from one folded away — a distinction the folded angle erased, letting an optimizer converge to the degenerate mirrored branch of a target. The face winding of the mesh is now treated as data, like the edge direction in `EdgeAngleGoal`: it must be unified (`compas.datastructures.Mesh.unify_cycles`) for the averaged vertex normal, and hence the signed angle, to be meaningful; no runtime check is performed. The winding tests in `test_goal_angle.py` flipped from invariance to covariance, and new tests pin the signed up-versus-down tangent contract and that optimization drives the tangent toward a signed target on its own branch.
- Bumped the COMPAS pin from `compas<2.0` to `compas>=2.15,<3.0` and migrated the codebase to the COMPAS 2.x API: `OptimizationRecorder` now serializes via `__data__`/`__from_data__`, the engine's update helpers use `index_node()`/`index_vertex()`/`index_edge()` instead of the removed `index_key()`/`index_uv()` (`FDMesh` grows an `index_edge()` to match `Graph`), and `mesh.edge_faces((u, v))` takes an edge tuple. The moved imports and renamed calls in the test suite were updated, and the committed JSON fixtures re-serialized into the 2.x format (they now require declaring source class).
- Raised the `requires-python` ceiling from `<3.12` to `<3.13`, added the 3.12 classifier, and added a Python 3.12 runner to the CI build matrix.
- Migrated the example scripts to the COMPAS 2.x API: `Polyline.divide_polyline()` became `Polyline.divide()` in `examples/arch/arch.py`.
- Ported the animation examples to `compas_viewer`, following its dynamic-drawing pattern: the frame callback goes through `@viewer.on(interval, frames)` (the old `timeout`/`record` kwargs are gone), mutates the datastructures in place and refreshes each scene object with `update(update_data=True)`.
- `OptimizationRecorder` grew a `__len__` so `len(recorder)` reports the number of recorded frames.
- An animated mesh redraws its shaded "Faces" surface alongside the wireframe on `obj.update()`.
- Repointed the `viz` extra from the discontinued `compas_view2==0.7.0` to `compas_viewer>=1.0`, making `pip install jax_fdm[viz]` resolvable again. The extra also grew `compas_notebook>=0.11` with the notebook port, plus `pythreejs` since the notebook scene objects import it directly to build batched render buffers. `numpy` moved into the explicit core dependencies for the same reason (it was only a transitive dependency through jax and compas before).
- Ported the 3D `Viewer` onto `compas_viewer`. The wrapper subclasses `compas_viewer.Viewer` and only keeps `add` as a convenience shortcut; the `width`/`height`/`show_grid` kwargs fold into a `compas_viewer.config.Config`.
- The custom viewer `Arrow` shape moved to `jax_fdm/visualization/shapes.py` so every visualization backend can share it rather than having a local copy.
- Set the viewer's default render mode to `"lighted"` and bumped the `Arrow` mesh resolution default from `u=4` to `u=8`. Also fixed the arrow head detaching from its body under COMPAS 2.x.
- `Arrow` conforms to the COMPAS 2.x `Shape` protocol: it implements `compute_vertices()`/`compute_faces()` (so the inherited `vertices`/`faces`/`edges` properties work), honors `resolution_u` (default 8), and `to_vertices_and_faces(triangulated=False, u=None, v=None)` matches the base signature.
- Changed the default node/vertex color from white to light gray.
- The scene objects now style their points in the vocabulary of their datastructure: `FDMeshObject`/`ThreeFDMeshObject` take `vertexcolor`, `vertexsize` and `show_vertices`, while `FDNetworkObject`/`ThreeFDNetworkObject` keep `nodecolor`, `nodesize` and `show_nodes`. Passing the node vocabulary to a mesh no longer works — and, per the scene-object protocol, is ignored silently rather than raising — so update mesh calls accordingly (the network API is unchanged).
- Grouped the viewer's `FDNetwork` elements into one foldable scene-tree entry, instead of hundreds of loose scene objects. Visibility toggles per network or per category.
- Batched each element category (edges as cylinders, nodes/vertices as spheres, load and reaction arrows) into a single triangle soup, built by vectorized numpy template tessellation (`visualization/buffers.py`). With `fuse=True`, a datastructure costs a handful of scene objects instead of two per element, and an animation frame updates one render buffer per category in place instead of re-adding hundreds of scene objects — `examples/animation/animation_mesh.py` drops from ~30 s to milliseconds per frame. The fused soup topology is fixed at add time: arrows are allocated for every candidate node/vertex and collapse to a degenerate soup while below tolerance (the per-element default prunes below-tolerance arrows at add time instead). With `fuse=True` selection granularity is per category, not per element.
- Swapped the viewer's buffer manager for a `FastBufferManager` that locates an object's slice in the combined render buffers with a numpy lookup instead of compas_viewer 2.0.2's per-vertex Python scan, keeping in-place buffer updates cheap for large scenes.
- The scene objects freeze the point-edge adjacency at add time, alongside the already-frozen soup topology and arrow candidate lists. compas's `Mesh.vertex_edges()` scans every mesh edge per call, which made the load-clearance and reaction-force gathering O(V·E) on every animation frame — the dominant per-frame cost (~32 of ~35 ms/frame on the monkey saddle). With the cached adjacency an update takes ~4 ms/frame.
- The 3D `Viewer` defaults changed from a 1280x720 window with the grid shown to a 1200x800 window without the grid, so `Viewer()` with no arguments fits a typical laptop screen and the grid does not cut through structures hanging below `z=0`. Pass `show_grid=True` to bring the grid back. The example scripts drop their now-redundant window arguments.
- The scene objects import the styling functions and constants of `visualization/style.py` one name per line, instead of through the `style` module namespace. Members of the scene-object contract went public: `build_soup()` (the fused/per-element parity surface), `recompute()` (re-deriving style state from the live datastructure), the frozen `adjacency`, `load_points` and `reaction_points` topology, and the raw style inputs, renamed with a `_spec` suffix (`edgecolor_spec` et al.) to keep them apart from the computed per-element dicts and the plain floats `compas_viewer` owns (`pointsize` et al.). Methods that override upstream hooks or only serve construction stay private.

### Fixed

- Repaired malformed numpy docstrings surfaced by the strict docs build: `EquilibriumModel`'s Parameters section used backticked names with no separator (unparseable by griffe), the Gauss-Newton and Levenberg-Marquardt solver docstrings named parameters `f`/`a` that don't exist in their `fn`/`theta` signatures, and `Parameter` documented `bound_up` as `bound_ip`.
- `angle_vectors` clips the cosine fed to `arccos`, whose value and gradient are singular when the vectors are parallel: in floating point the cosine of two parallel vectors can overshoot 1 by a few ulps, and the angle came out `nan`. The fix lives in the geometry function so every angle consumer is guarded at once — `EdgeAngleGoal`, `EdgeAngleConstraint`, the vertex angle goals and `angles_polygon` all funnel through it. New regression tests pin the parallel and antiparallel cases at the geometry level and on `EdgeAngleGoal` with edges exactly parallel to the reference vector.
- Fixed `viewer.add(fdnetwork, name="...")` ignoring the `name` keyword for an `FDNetwork` or `FDMesh`. An explicit `name=` now takes precedence over the datastructure's own name in the scene tree.
- The `Viewer` clamps the requested window size to the available screen geometry, silencing Qt's "window position outside any known screen" warning when a script asks for a window larger than the display. The clamp prints a warning with the resized window dimensions.
- Fixed pytest collection crashing without the `viz` extra installed: the root `conftest.py` still ignored the deleted artist-era module names, so `--doctest-modules` imported the new scene-object modules and died on their `compas_viewer`/`pythreejs` imports. The viewer and notebook subpackages are now ignored wholesale when their backend is missing. CI installs `.[dev,viz]` (plus the system GL libraries PySide6 needs on the Linux runners), so the viewer tests run instead of skipping.
- Fixed the freeze when visualizing several results in a row in one process (e.g. one per step of a sequential optimization). `compas_viewer.Viewer` is a process-wide singleton whose `running` flag survives closing the window, so every `scene.add` after the first `show()` triggered a full buffer and scene-tree rebuild over the accumulated objects (quadratic slowdown, or a crash against the dead GL context). The supported pattern is now to reuse one `Viewer` across shows: the new `viewer.clear()` empties the scene (including the picking-color registrations the parent never prunes) between shows, and `show()` rebuilds the GL render buffers on a re-show and resets `running` on return, so between-show adds stay lightweight. Sequential scripts construct the viewer once and run `clear()`/`add()`/`show()` per step; the camera carries over between steps.

### Removed

- Removed the compas 1.x plotter artists (`FDNetworkPlotterArtist`, `FDVectorPlotterArtist`, `FDVector` and their `Artist.register` calls). They targeted the discontinued bundled `compas_plotters` and could no longer import; the scene objects of the 2D plotter port replace them, and `examples/arch/arch_plotter.py` is back on the new backend.
- Removed `jax_fdm/visualization/notebooks/register.py`: the compas 1.x `Artist.register` factory it targeted no longer exists. Scene registration now lives in the per-backend `scene_objects.py` modules and runs from the backend `__init__` when the backend is installed; the mesh scene objects pin the native mesh scene object via `sceneobject_type` to avoid recursing into the registered force density type.
- `viewer.add` now dispatches purely by type, the `as_wireframe`/`as_mesh` flags were eliminated.
- Removed the artist layer (`jax_fdm.visualization.artists` and the per-backend `*Artist` classes) in favor of the registered scene objects. `viewer.artists` is gone: hold the scene object returned by `viewer.add(...)` and call `obj.update()` in animation loops.

## [0.11.0] 2026-07-04

### Added

- Added `tests/test_adjoint_routing.py`, which pins *which* differentiation path each model configuration executes (linear solve at `tmax=1`, implicit adjoint at `tmax>1` with `implicit_diff=True`, unrolled autodiff otherwise) by spying on the routing functions rather than the numbers. The gradient-correctness tests pass regardless of the path taken, so these guard against a refactor silently rerouting the backward pass. The adjoint path is covered for both dense and sparse stiffness matrices, and the assertions are execution-only, so they stay backend- and precision-independent.
- Added a COMPAS-free test suite under `tests/` that pins the force density engine behavior (solver, optimizer, recorder, connectivity, load assembly) ahead of the COMPAS 2.x migration. The tests are invariant-first, so only two small "golden" files are committed. COMPAS cross-checks are quarantined behind a `compas_xcheck` marker.
- Added `tests/test_reference_solver.py`, cross-validating the solver against an independent structural-analysis solver (Stiff3D by TUM): two self-stressed reference networks from committed CSV fixtures are recovered to within `1e-8`.
- Added `tests/test_optimizer_parameters.py`, covering previously untested nodal load parameters (to `1e-9`) and grouped force-density, support, and load parameters across all three axes. The load sweep also guards the Y-component regression.
- Added optional dependency extras to the package metadata: `viz` (`compas_view2`), `ipopt` (`cyipopt`), and `dev`.
- Added a `.pre-commit-config.yaml` with `ruff` and the standard whitespace, YAML, TOML, and merge-conflict checks, and added `pre-commit` to the `[dev]` extra.
- Added `tests/test_arch_benchmark.py`, validating the optimizer against the closed-form analytical arch benchmark in Pastrana et al. (2026), *Differentiable force density method for the design of lightweight structures* (doi:10.1016/j.cma.2026.118783), Appendix A.1. The optimized rise and load path match the analytical values to the paper's reported accuracy (rise within 0.1%, load path within 1% by `n_v=100`). The benchmark originates with Liew (2020) (doi:10.1016/j.istruc.2020.09.078).
- Added `tests/test_taylor_convergence.py`, validating the analytical adjoints from Pastrana et al. (2026), Appendix A.4, with the Taylor remainder test. Both the linear-solver and URS adjoints are checked on compact generated geometry, matched against an unrolled autodiff gradient. There is also a corrupted-gradient test for when things go south.
- Added `tests/test_liew_dome.py`, validating the optimizer against the circular gridshell from Liew (2020), *Constrained Force Density Method optimisation for form-finding* (doi:10.1016/j.istruc.2020.09.078), Section 4.2. The 613-vertex dome is committed as a frozen fixture so ordering stays fixed for order-sensitive SLSQP. The base case matches the paper exactly, and the volume-only optimization reproduces its statistics. scipy's 1.16.0 Fortran-to-C SLSQP rewrite changed convergence on this problem (exit mode 0 before, exit mode 4 after), so the optimization test is gated with `skipif` to scipy < 1.16.0 and marked `slow`.
- Added `tests/test_goal_angle.py`, guarding the shape contract of angle goals: it drives constrained form finding with multiple `EdgeAngleGoal`s (the multi-edge path that regressed) and pins that the goal's prediction shape matches its target. The test fails against the pre-fix scalar `angle_vectors`. The vertex angle goals (`VertexNormalAngleGoal`, `VertexTangentAngleGoal`) are additionally checked for correctness and `nan`-reliability on a ragged tri + quad mesh, where a new `ragged_mesh` fixture triangulates two meshgrid faces so the shorter `faces_indexed` rows are `-1`-padded. On a fixed-geometry mesh the normal angle is exactly 0 (flat) and exactly `theta` (plane tilted by `theta`, checked at triangle-incident vertices), matching the all-quad baseline. A ragged optimization run then confirms the padding-replacement `jnp.where` backpropagates `nan`-free, and a whole-mesh normal-angle `MeanAbsoluteError` loss is checked to have a finite gradient.

### Changed

- Capped `requires-python` to `>=3.10,<3.12` to match the supported and tested Python range. The `compas<2.0` dependency targets 3.10/3.11, the classifiers already listed only those, and the CI matrix runs only 3.10/3.11, so the previous open-ended `>=3.10` advertised untested 3.12+ installs.
- Decoupled the equilibrium engine from COMPAS by inlining the helpers it borrowed: `connectivity_matrix` into `equilibrium/structures/graphs.py`, `face_matrix` into `meshes.py`, and the `pairwise` iteration folded directly into its two mesh call sites. The core solver (`equilibrium/`, `goals/`, `constraints/`, `losses/`, `parameters/`, `geometry/`) no longer imports COMPAS at all; COMPAS stays an optional dependency for the `FDNetwork`/`FDMesh` I/O and construction layer. The inlined code reproduces the COMPAS 1.x semantics exactly and is pinned by the `test_connectivity.py`/`test_loads.py` characterization tests, which are now COMPAS-version-agnostic.
- Stopped passing the `disp` option to scipy's `L-BFGS-B` in `LBFGSB.options()`. scipy 1.16 dropped `disp` from that method's accepted options (with no options-based verbosity control left in its place), so every solve raised `OptimizeWarning: Unknown solver options: disp`. `LBFGSB` already defaulted `disp` to `False` and the optimizer prints its own progress, so nothing is lost. Other methods (SLSQP, BFGS) still accept `disp` and are unaffected.
- Replaced the deprecated `lineax.NormalCG(...)` in the implicit-differentiation adjoint solver (`equilibrium/solvers/fixed_point.py`) with `lineax.Normal(lineax.CG(...))`, its exact equivalent, silencing the deprecation warning and guarding against its removal in a future Lineax release.
- Migrated packaging from `setup.py`/`setup.cfg` to `pyproject.toml` with the setuptools build backend. `src`-rooted package discovery fixes a long-standing bug where `packages=["jax_fdm"]` shipped only the top-level package and omitted every subpackage from the wheel. Tool configuration for `ruff` (replacing `flake8`/`isort`/`pydocstyle`) and `pytest` now lives in `pyproject.toml`, and the bogus `bdist_wheel universal=1` `py2.py3` tag is gone.
- Made the optional visualization backends degrade when absent, via a new `visualization/backends.py`. Missing backends no longer break `import jax_fdm.visualization`, and `Viewer`/`NotebookViewer` become inert null objects that warn once on construction. `LossPlotter` works unconditionally.
- The package `matplotlib` was promoted to a core dependency.
- Patched `numpy.int=int` in `visualization.viewer` so `compas_view2=0.7.0` keeps working with `numpy>=1.24` (which deprecated `numpy.int`), since `jax` requires modern numpy. Remove this once we migrate to `compas>2` and `compas_viewers`.
- Cached `FDNetworkArtist.node_xyz` instead of recomputing it for every drawn edge, cutting plotting time by an order of magnitude (an upstream `compas.artists` bug in the legacy version we use).
- **Breaking:** renamed `NodeNormalAngleGoal`/`NodeTangentAngleGoal` to `VertexNormalAngleGoal`/`VertexTangentAngleGoal` and moved them from `goals/node/` to `goals/vertex/`. These goals compute a vertex normal from the faces surrounding a vertex, so they only ever worked on a mesh; the `Node*` names implied network support that never existed. Their internals were also rewritten onto the current mesh primitives (`structure.faces_indexed` + `structure.connectivity_faces_vertices`), replacing the removed `connectivity_faces`/`face_node_index` accessors and the hand-built nan-masked polygon machinery. The new path averages incident face normals with the gradient-safe `-1`-padding trick used by `MeshAreaGoal`, so it no longer risks nan gradients on ragged faces.
- `Optimizer` now retains the backend result of its last solve on `self.result` (the scipy `OptimizeResult`), exposing the convergence status, message, and multipliers that `solve` previously discarded. `constrained_fdm` still returns the equilibrium network unchanged.
- Added Windows to the `build.yml` test matrix: Python 3.10 now runs on Ubuntu, macOS, and Windows plus one Ubuntu job on 3.11, covering the supported floor and ceiling within the previous four-job count.
- Reworked `examples/pillow/pillow.py` to drop `compas_singular` and run natively on an `FDMesh`: the grid comes from `FDMesh.from_meshgrid`, curvature-constrained nodes are selected by vertex index, and the FDM pipeline consumes the mesh directly (no `FDNetwork`, except to render force-colored edges in the viewer).
- Reworked `examples/monkey_saddle/monkey_saddle.py` and `monkey_saddle_constraints.py` to drop the `compas_singular`/`compas_quad` dependency and run natively on an `FDMesh`: densification uses stock `mesh_subdivide_quad` (still density-driven via subdivision level `k`), anchors are selected by splitting the boundary loop at its corners instead of via polyedges, and the FDM pipeline consumes the mesh directly (no `FDNetwork`, except to render force-colored edges in the viewer). In the constrained variant, the edges between two anchored vertices are now pinned out of the optimization instead of removed from the network.

### Removed

- Removed the sequential form-finding examples `examples/dome/dome_sequential.py` and `examples/pringle/pringle_sequential.py` (shipped through 0.10.0 as `dome_temporal.py`/`pringle_temporal_horizontal.py`).
- Removed `NodeNormalAngleConstraint`/`NodeTangentAngleConstraint`. Like the angle goals, they computed a vertex normal from surrounding faces (so they only worked on a mesh) and read the removed `connectivity_faces`/`face_node_index` accessors, but unlike the goals they had no callers anywhere in the library, examples, tests, or docs. They were deleted rather than moved to the vertex family; add a `VertexNormalAngleConstraint` on demand if a use case appears.
- Removed the `if __name__ == "__main__"` blocks from `loads.py`, `meshes.py`, and `graphs.py`. Their ad-hoc checks are now real tests in the pytest suite.
- Removed `MANIFEST.in`, now redundant with `src`-rooted package discovery and `pyproject.toml` metadata.
- Removed `.bumpversion.cfg`. Moved its configuration to `[tool.bumpversion]` in `pyproject.toml` and dropped the obsolete `setup.py` and Sphinx `docs/conf.py` file targets.
- Slimmed `tasks.py`: dropped the dead Grasshopper and Sphinx tasks; `lint` now runs `ruff` and `release` uses `bump-my-version`.
- Modernized the GitHub Actions workflows: `build.yml` installs with `pip install -e ".[dev]"` and runs `ruff`/`pytest` directly (replacing `compas-actions.build`); `release.yml` drops `::set-output` and `actions/create-release@v1` for `softprops/action-gh-release`, `python -m build`, and `pypa/gh-action-pypi-publish`; and `checkout`/`setup-python` are bumped across all workflows.
- Modernized `.editorconfig`: dropped the dead `*.bat`/`*.cmd`/`*.ps1` and `Makefile` blocks, added a `*.toml` rule, broadened the YAML glob, and switched the project URL to HTTPS.
- Enabled ruff's `W` (pycodestyle warnings) rule set, enforcing the `.editorconfig` whitespace conventions in the linter.
- Sorted and grouped imports across the source tree to satisfy ruff's `I` (isort) rule.
- Removed the `conda_osx.yml` and `conda_linux.yml` environment files: byte-for-byte identical, unreferenced, and stale. The `conda`-only dependencies are documented in the README.
- Consolidated dependency declarations into `pyproject.toml` (runtime deps inline under `[project.dependencies]`, dev deps in the `[dev]` extra) and removed `requirements.txt`/`requirements-dev.txt`.
- Widened the CI lint step from `ruff check src` to `ruff check .`, applying import/whitespace fixes across `examples/`, `docs/`, and `tests/` and adding `per-file-ignores` for `src/jax_fdm/__init__.py` and `src/jax_fdm/equilibrium/states.py`.

### Fixed

- Removed the `assert optimizer.result.status == 0` check from `test_liew_dome.py::test_volume_optimization_matches_paper`. On the `macos-latest` CI runner (arm64, Apple Accelerate BLAS) SLSQP stalls in linesearch near the optimum and reports exit mode 8 ("positive directional derivative"), even though the solution still matches the paper within tolerance; the strict exit-code assertion is too brittle across CI runners for this borderline problem. Validation is now purely metric-based (volume, lengths, and deviation within tolerance), which still catches a genuinely wrong result. The `skipif` to scipy < 1.16.0 stays, since scipy >= 1.16 diverges to a different, wrong optimum.
- Fixed `NodeGroupLoadYParameter` and `VertexGroupLoadYParameter`, which inherited from the `X` parameter and so parametrized `px` instead of the Y component. They now inherit from the matching `Y` parameter. Also corrected the misspelled `VertesGroupLoadYParameter`.
- Fixed a shape regression in the angle goals (`EdgeAngleGoal`, `VertexNormalAngleGoal`, and thus `VertexTangentAngleGoal`; then still named `NodeNormalAngleGoal`/`NodeTangentAngleGoal`). When `angle_vectors` was refactored to delegate to `cosine_vectors`, its output lost the trailing axis and became a scalar, so a goal's prediction came out `(N,)` while the `ScalarGoal` target stayed `(N, 1)`, raising an assertion in `Goal.__call__`. The goal predictions now wrap `angle_vectors` in `jnp.atleast_1d`, restoring the `(N, 1)` shape; `angle_vectors` stays scalar so `angles_polygon`/`FaceRectangleGoal` are unaffected.
- Fixed `VertexNormalAngleGoal` (and thus `VertexTangentAngleGoal`) chasing the wrong angle branch. The prediction took the angle from the *signed* cosine of the vertex normal against the reference vector, so it depended on face winding: a mesh whose faces wound the averaged vertex normal downward reported `pi - theta` instead of `theta`. A downward-pointing normal made the tangent goal (`pi/2 - angle_normal`) start negative and unable to reach a positive target, so the optimizer flattened the surface to a zero tangent angle instead of converging. The prediction now takes the angle from the *absolute* cosine, folding it into `[0, pi/2]` so it is invariant to the normal's orientation, and clips the `arccos` argument to guard its gradient singularity when the vectors align. `test_goal_angle.py` gains winding-invariance and upper-hemisphere regression tests that fail against the signed-cosine version.


## [0.10.0] 2026-05-07

### Added

- Added journal paper citation to `README.md`.
- Implemented `FaceRectangularGoal`, the first face-based goal in this library to promote equilibrium solutions where every edge in a quad face is orthogonal to its two neighboring edges.
- Introduced `goals.FaceGoal()` object to define goals on mesh faces!
- To support the point below, we added `MeshIndexingMixins.face_index` as a property precisely as a way to map between an `FDMesh.faces()` and `structures.Mesh.faces`.
- Added `Mesh.face_keys` as a datastructure attribute to bookkeep the face keys (fkeys) of an `FDMesh`. Storing such keys is needed to support goals defined on mesh faces while assigning them via fkeys through the `Goal` interface. This feature allows a 1:1 info mapping from an `datastructureFDMesh` to a `structures.Mesh`. A bit hacky, if you ask me, but it works.
- Implemented `cosine_angles_polygon()`, a function that measures the internal angle cosines of a polygon (i.e., the dot product between outgoing sides at every polygon vertex).
- Implemented `angles_polygon()`, a function that measures the internal angles of a polygon.
- Extracted logic to `cosine_vectors()` to measure the cosine of the angle between two vectors.
- Implemented `goals.NodesCurvatureGoal()` to control the curvature of a sequence of nodes.
- Implemented `goals.NodesColinearGoal()` to make a sequence of nodes colinear.
- Added `curvature_points()` to measure the discrete curvature of a sequence of points.
- Created `colinearity_points()` to measure the colinearity of a sequence of points.
- Implemented `goals.MeshLoadPathGoal()` to control the total load path enery of a mesh.
- Implemented `goals.MeshPlanarityGoal()` to planarize all the faces of a mesh.
- Added `polygon_planarity()` to geometry processing module (with tests!). The planarity of a polygon is calculated as the sum of the absolute dot product between the polygon's unitized normal vector and its unitized edge vectors, following the work of Tang et al. (2014).

### Changed

- Fixed bug in `line_lcs()` and `polygon_lcs()`, functions that calculate the local coordinate system of a line and a polygon, respectively. The bug was that while the frame normal was properly unitized, the other two vectors were not.
- Refactored `LogMaxError()` for numerical stability. Specifically, replaced `jnp.log(x+1)` with `jnp.log1p` and `jnp.where` with `jnp.maximum`.
- Fixed bug in `normalized_vector()` that returned a vector of ones when supplied a zero vector. This is a special case because of the undefined behavior of division by zero. After the fix, we decided that the function should return the zero vector if one such vector is input to the function.

### Removed

- Updated docstring of `normalized_vector()` to declare the function is not fully `nan` safe, even though it internally performs `nansum` to sum over cross-products. The problem is that information is lost in that sum, and the resulting normal vector is erroneous.

## [0.9.0] 2026-01-06

### Added

- Implemented a vanilla `GradientDescent` optimizer.
- Implemented `splu_cpu` and `splu_solve` to cache sparse stiffness matrix factorization using scipy's sparse infrastructure. These functions specifically instantiate `scipy.sparse.linalg.splu`, a sparse LU factorization using the `SuperLU` package under the hood. We implemented both these functions by wrapping them into `jax.pure_callback` for compatibility with `jax.jit`.
- The `fixed_point_bwd_adjoint` of the fixed point solver `solver_fixedpoint_implicit` now caches a sparse stiffness matrix to avoid refactorizing it every time the a linear solve is invoked by `lineax` to resolve the adjoint system. Caching reduces computation cost considerably (about 30% in the British Museum example).
- Created `GradientFreeOptimizer()` class for other gradient-free and evolutionary algorithms to subclass it. This new parent class implements specific `problem()` and `solve()` methods that accounts for the different naming convention of the objective function in `scipy.minimize` (`fun` vs. `func`). Moreover, the new method does not calculate loss and gradients per iteration, only the loss, since the gradient is unnecessary for gradient-free optimization. Omitting the gradient calculation and focusing on the loss alone speeds up optimization wall time.
- Added attribute `EquilibriumModel.iterload_fn` to host a callable that modifies a load state before starting iterative FDM. This is our current alternative to the previous option that zeroed out the vertex/node loads without a warning. It also opens the door to other types of load changes between the first and second step of form-finding, giving users more flexibility. Modified the functions in `fdm.py` by exposing `iterload_fn` as an argument. These functions are `model_from_sparsity()`, `fdm()`, and `constrained_fdm()`.
- Added `Viewer.save()` to automatically save images of the current viewer scene.
- Added automatic support for dense and sparse stiffness matrices in `custom_vjp` of `solver_fixed_point_implicit()`. For sparse matrices, we apply `jax.lax.custom_linear_solver()` as a thin wrapper around the sparse linear solve defined in `EquilibriumModel.linearsolve_fn()` to generate a transpose rule for it. The transpose rule is required by `lineax`, inside `FunctionLinearOperator`. Without the wrapper and the transpose, we cannot use implicit differentiation with a sparse linear solver and a fixed-point solver. Now we can.
- Implemented `geometry.length_vector_sqrd()`.
- Pass `implicit_diff` argument to `solver_fixedpoint`.
- Print out statistics with `ndigits` of precision in `FDDatastructure.print_stats()`.
- Listed `lineax` and `optimistix` as dependencies.
- Added `EquilibriumModel.residual_free_matrix()` to compute the matrix with the residual force vectors on the free vertices of a structure.
- Set up a `custom_vjp` with the implicit function theorem for the nonlinear equilibrium solvers (least squares and root finding).
- Wrapped up 3 different `optimistix` optimizers to solve the nonlinear equilibrium problem with shape dependent loads. These solvers are least-squares and root-finding optimizers: `Newton`, `Dogleg`, and `LevenbergMarquardt`. These solvers minimize the residual function explicitly, which differs from the fixed-point iterators that solve the equilibrium problem by minimizing the difference between the XYZ coordinates of the free vertices of a structure over two consecutive iterations. These solvers are listed in the API as `solver_newton`, `solver_dogleg`, and `solver_levenger_marquardt`.
- Exposed the `maxcor` argument in scipy's `LBFGSB()` wrapper. This argument controls the number of approximation terms of the full Hessian.
- Added `error_terms` argument in `LossPlotter.plot()` to select what error and regularization terms are plotted.
- Implemented `EquilibriumModel.load_xyz_matrix` to calculate the load matrices for shape dependent loads.
- Implemented `EquilibriumModel.load_xyz_matrix_from_r_fixed` to calculate the load matrices for shape dependent loads.
- Added `is_solver_fixedpoint` and `is_solver_leastsquares` to check the type of an iterative solver.
- Added `solver_gauss_newton` to calculate equilibrium states in the presence of  shape-dependent loads by minimizing a residual function explicitly. This solver is wrapped up from `jaxopt`.
- Exposed `report_breakdown` argument in `LossPlotter.plot()` to optionally plot the contributions of the error and regularization terms of a `Loss` function.
- Implemented `Optimizer.options()` to allow for method-specific setup in `scipy.optimize.minimize`. This new method assembles the `options` dictionary required by `scipy` in a way that can be customizable per optimizer.

### Changed

- Renamed `network` argument input to `Optimizer.solve()` to `datastructure` for correctness. This method can both ingest an `FDNetwork` or an `FDMesh`.
- The callback function is called once before optimization starts in `Optimizer.solve()` and in `GradientFreeOptimizer.solve()`.
- Fixed bug that forgot to fix the random seed of `DualAnnealing()` despite being passed in as an argument at initialization.
- Stop passing `EquilibriumModel.linearsolve_fn()` as `solver_kwargs` of fixed point solver. This function was used only for the `custom_vjp` operations. The `custom_vjp` now selects and appropriate linear solver based on whether `EquilibriumModel.stiffness_matrix()` is a sparse `jax.CSC` object or not.
- Now, `goals.NetworkSmoothGoal()` calculates the fairness energy on all the vertices. Previously, it only considered the free vertices.
- `FDDatastructure.print_stats()` doesn't reporting positive or negative force and force densities if the datastructure doesn't contain them.
- Changed the diagonal matrix generated by `EquilibriumStructureSparse._get_sparse_diag_data()` from `jax.experimental.sparse.CSC` to `jax.experimental.sparse.BCSR` for enabling Jacobian computations.
- Exposed arrow parameters (head width, head portion, body width and minimum width) in `FDVectorPlotterArtist`.
- Fixed bug in `FDNetworkViewerArtist()` that ignored custom colors when drawing node loads.
- `LossPlotter.print_error_stats()` reports the loss and error values with up 6 digits of precision.
- Renamed `EquilibriumModel.force_fixed_matrix()` to `EquilibriumModel.residual_fixed_matrix()`.
- Renamed `EquilibriumModel.force_matrix()` to `EquilibriumModel.load_matrix()`.
- `EquilibriumParametersState.from_datastructure` takes `dtype` as optional input. It defaults to `jax.numpy.float64`.
- Changed `DTYPE_INT` to `int64` instead of `int32`.
- To calculate the local coordinate system of a mesh face, `loads.face_load_lcs()` no longer replaces vertex indices that were padded with a `-1` with `face[0]`. Instead, it takes all the vertices in a `face` to get the XYZ coordinates of the face polygon. The previous behavior led to due excessive compilation time and XLA warnings due to "constant folding" problems because of the index replacement with a vmapped `jnp.where()`.
- ~~The faces generated by `EquilibriumStructureMesh.from_mesh` are now padded with first index `face[0]` instead of a `-1`.~~
- `EquilibriumModel` automatically picks the iterative equilibrium function based on solver input as `iterativesolve_fn`.
- Functions `EquilibriumModel.equilibrium()` and
`EquilibriumModel.equilibrium_iterative()` return `xyz_free` instead of `xyz`. The concatenation of `xyz_free` and `xyz_fixed` needed to build `xyz` is now handled by `EquilibriumModel.__call__()`.
- Set `implicit=False` and `unroll=False` in `solver_anderson` when performing implicit differentiation on iterative equilibrium calculation with `implicit_diff=True`.
- Set `implicit=False` and `unroll=False` in `solver_fixedpoint` when performing implicit differentiation on iterative equilibrium calculation with `implicit_diff=True`.
- `jax_fdm.equilibrium.datastructure_validate` now reports the number of edges with zero force densities.
- `LBFGSBS` became a subclass of `LBFGSB` instead of `Optimizer`.
- Disabled hard assertion test that ensured that every edge in a `topology.Mesh()` object was connected to at most 2 faces (manifoldness preservation). Now we print out a warning since we are all consenting adults over here. The implications of this change is that area load calculations might be incorrect, but this needs to be more thoroughly tested at a later time.

### Removed

- Removed attribute `EquilibriumModel.ignore_nodes_load`/`EquilibriumModel.nodes_loads_iter` because zeroing out node lodes behind the scenes can come bite you. Better to explicitly zero out them.
- Deleted `jax_fdm.loads._faces_load_2` because it was not longer used.

## [0.8.6] 2024-10-30

### Added

- Wrapped two gradient-free optimizers from scipy: Nelder-Mead and Powell. They are available as `jax_fdm.optimizers.NelderMead` and `jax_fdm.optimizers.Powell`, respectively.
- Linked two evolutionary optimizers from scipy They are available as `jax_fdm.optimizers.DualAnnealing` and `jax_fdm.optimizers.DifferentialEvolution`.
- Added support for kwargs in `LossPlotter.plot()`. The kwargs control the parameters of the equilibrium model used to plot the loss history.
- Added `VertexSupportParameter.index()`. This change might appear redundant, but it was necessary to deal with the method resolution order of the parent classes of `VertexSupportParameter`.
- Added `VertexGroupSupportParameter.index()` for similar reasons as the listed above.

### Changed

- Changed `datastructure.print_stats()` to report positive and negative forces separately.
- Turned off display in `TruncatedNewton`.
- Fixed bug in `OptimizationRecorder`. The recorder did not know how to record optimization history without an explictly initialized optimizer.
- Deprecated `jax_fdm.optimization.optimizers.scipy` in favor of `jax_fdm.optimization.optimizers.gradient_based`.
- Fixed bug. Return early in `NetworkArtist.edge_width()` if the artist edges list is empty.
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
