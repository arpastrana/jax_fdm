# Form-finding

Form-finding is the act of letting forces decide a shape.
You fix a few supports, hang some loads, declare how stiff each edge pulls, and then ask: what geometry holds all of this in equilibrium?
The force density method answers that question with a single linear solve, and JAX FDM wraps the whole affair in one function call.

This guide starts at the front door, the public `fdm` function on a network or mesh, and then opens the hood to show the JAX objects that do the work underneath.
When you later write a [custom goal](custom_goals.md) or [custom constraint](custom_constraints.md), you will be reaching straight into those objects, so it pays to meet them here first.

## Form-finding a datastructure

You model a structure as a COMPAS-flavored datastructure: an `FDNetwork` for bars and cables, an `FDMesh` for surfaces.
The datastructure carries everything the force density method needs, all as ordinary attributes you set before solving:

- a **force density** on every edge (the signed force-to-length ratio; negative pulls in compression, positive in tension),
- a set of **supports**, the nodes held fixed in space,
- the **loads** applied to the free nodes.

To make this concrete, let us form-find the oldest shape in the book: a **hanging cable**.
Pin a chain of nodes at both ends, hang a weight from each node in between, and let it sag into a catenary, the curve a slack rope settles into under gravity.
We build the network from scratch, six nodes in a straight line joined by five edges, so the whole example stands on its own with no data file:

```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm


nodes = [[x, 0.0, 0.0] for x in range(6)]
edges = [(i, i + 1) for i in range(5)]

network = FDNetwork.from_nodes_and_edges(nodes, edges)

network.edges_forcedensities(q=2.0)                               # every segment pulls in tension
network.nodes_supports([0, 5])                                    # pin the two end nodes
network.nodes_loads([0.0, 0.0, -1.0], keys=network.nodes_free())  # hang a unit weight from the rest

eq_network = fdm(network)
```

That is a complete form-finding run.
Every segment gets a positive force density of `2.0` (tension pulls the cable taut), the two end nodes are pinned as supports, and each interior node carries a downward unit load.
The straight line you fed in comes back sagged into a symmetric curve: `eq_network` is a new datastructure whose interior nodes have dropped to their equilibrium heights and whose edges now report their solved `length`, `force`, and force density `q`.
The input `network` is left untouched, so you can form-find the same model many times over.

!!! tip "Flip it for an arch"

    Change the force densities from `+2.0` to `-2.0` and the same script gives you a compression arch instead of a tension cable, the catenary turned upside down.
    That inversion, tension and compression as mirror images, is the whole reason funicular form-finding is such a powerful design tool.

!!! note "The result is a copy"

    `fdm` never mutates its input.
    It reads the force densities, supports, and loads off the datastructure you hand it, solves for equilibrium, and writes the solved geometry into a fresh copy.
    Keep the return value; the original stays as you built it.

### Kinds of load

The example above loads nodes directly, but a structure can be loaded on three different elements.
Whatever the element, the force density method ultimately needs a load *at each node*, so edge and face loads are distributed to the nodes they touch by tributary geometry:

| Load | Applies to | How to set it | Distributed to nodes by |
| --- | --- | --- | --- |
| **Node / vertex** | a node (`FDNetwork`) or vertex (`FDMesh`) | `nodes_loads`, `node_load` / `vertices_loads`, `vertex_load` | applied directly, no distribution |
| **Edge** | an edge | `edges_loads`, `edge_load` | split to the edge's two end nodes by tributary length |
| **Face** | a face (`FDMesh` only) | `faces_loads`, `face_load` | split to the face's vertices by tributary area |

A node or vertex load is a plain point load: whatever you set is what the node carries.
Edge and face loads are **line** and **area** loads, magnitudes per unit length or area, that the model resolves onto nodes for you.

!!! warning "Edge and face loads are shape-dependent"

    A point load on a node stays put, but the nodal share of a line or area load depends on the current edge lengths and face areas, which change as the structure moves.
    So edge and face loads only take effect when the iterative solver is on: at the default `tmax=1` they are discarded (see [tuning the solve](#tuning-the-solve)).
    Raise `tmax` to let the solver settle the shape and its loads together.

### Tuning the solve

`fdm` takes a handful of keyword arguments that shape *how* equilibrium is computed.
The defaults suit most problems, so reach for these only when you need them:

- **`sparse`** (default `True`) picks the sparse solver, which scales to large structures. More on this in [the numerical core](#the-numerical-core) below.
- **`tmax`** (default `1`) is the maximum number of equilibrium iterations. With `tmax=1` the model takes a single linear force density step, which is the classic FDM and all that most models ever need. Raising it turns on an iterative solve for **shape-dependent loads**, loads that must be recomputed as the geometry moves (self-weight, area loads on a mesh, edge line loads).
- **`is_load_local`** (default `False`) applies edge and face loads in their own local frames as the shape changes (follower loads), rather than keeping them fixed in the global frame.
- **`eta`** (default `1e-6`) is the convergence tolerance for that iterative solve.

### The constrained sibling

`fdm` finds *an* equilibrium, whichever one your force densities happen to produce.
Often you want a *particular* equilibrium: the shape whose edges stay under a meter, or the one that minimizes load path.
That is the job of `constrained_fdm`, which wraps `fdm` in an optimization loop.
It is a whole workflow of its own, so it gets its own guide: [constrained form-finding](constrained_form_finding.md).

## The numerical core

Everything above happens on the datastructure, in the comfortable world of node keys and edge tuples.
But the actual solve happens in JAX, on plain arrays.
Between the two, `fdm` quietly assembles a small cast: a **structure** and a **parameter state** as inputs, a **model** that maps one to the other, and an **equilibrium state** as output.

```
model(parameters, structure) -> EquilibriumState
```

You never build these by hand for a basic solve, yet they are exactly what your custom goals and constraints read from, so here is what each one is.

### The structure

A **structure** is the datastructure's topology, frozen into a form JAX can differentiate through.
An `FDNetwork` or `FDMesh` is a rich, mutable COMPAS object: a dictionary of nodes and edges keyed by arbitrary integers, carrying attributes, geometry, and helper methods.
That is wonderful for modeling and terrible for a numerical core, which wants contiguous arrays with fixed shapes and no Python-level lookups on the hot path.
The structure is the bridge: `fdm` converts your datastructure into one before solving.

The conversion (`EquilibriumStructure.from_network`, `EquilibriumMeshStructure.from_mesh`) does two things.
First, it **numbers the elements**: it walks `network.nodes()` and `network.edges()` in generator order and assigns each a contiguous integer index, building the **index tables** (`node_index`, `edge_index`) that map a key to its row. This is where the deterministic ordering comes from, the first edge yielded is row `0`, the last is row `number_of_edges() - 1`.
Second, it **precomputes the connectivity** as matrices: the signed edge-node incidence matrix, split into free and fixed submatrices via the support mask, plus the index maps that reorder nodes between their native and free-fixed arrangements. These are the matrices the force density method multiplies together, assembled once so the solve never has to touch the datastructure again.

The structure is an immutable [equinox](https://docs.kidger.site/equinox/) module, which means JAX treats it as a registered pytree: its static index arrays stay as NumPy (topology never needs a gradient), while its connectivity matrices are JAX arrays that the differentiable solve flows through.

This is the object behind the key-versus-index duality you will meet in the [goals](goals.md#keys-versus-indices) guide.
The structure records the numbering once, deterministically, so every later lookup, every goal's `init`, every parameter's index, agrees on which row is which element.

Structures come in four flavors, one per datastructure kind and solver: network or mesh, dense or sparse.
`fdm` picks the right one for you from the datastructure you pass and the `sparse` flag.

### The parameter state

If the structure is the fixed topology, the **parameter state** is the variable data the solve is a function of, the independent quantities that, together with the connectivity, pin down one equilibrium.
An `EquilibriumParametersState` is a named tuple of three arrays, read off your datastructure by `EquilibriumParametersState.from_datastructure`:

| Field | Shape | What it holds |
| --- | --- | --- |
| `q` | `(edges,)` | the force density of every edge, in structure order |
| `xyz_fixed` | `(nodes_fixed, 3)` | the coordinates of the supported nodes |
| `loads` | a `LoadState` | the applied loads, split by element |

The `loads` field is itself a small `LoadState` named tuple, because loads can enter on three different elements:

| Field | Shape | What it holds |
| --- | --- | --- |
| `nodes` | `(nodes, 3)` | the load vector applied directly to each node |
| `edges` | `(edges, 3)` or `0.0` | the line load on each edge |
| `faces` | `(faces, 3)` or `0.0` | the area load on each face (meshes only) |

Edge and face loads collapse to the scalar `0.0` when every entry is zero, which lets the model skip distributing them altogether; a network always carries `faces=0.0`.

These arrays are the reason the datastructure conversion matters: `q` is ordered by the structure's edge numbering, `xyz_fixed` by its support numbering, so the parameter state and the structure speak the same index language.
For a plain `fdm` call these come straight off your datastructure; under `constrained_fdm`, they are exactly what the optimizer is free to change, one number at a time.

### The model

A **model** is the force density method itself, expressed as a differentiable function.
Give it the parameter state and the structure and it returns the equilibrium state:

```
model(parameters, structure) -> EquilibriumState
```

There are two models, and they compute the same equilibrium by different means:

- `EquilibriumModel`, the **dense** model, assembles the stiffness matrix as a full array and solves the linear system directly. Simple and fast for small to medium structures.
- `EquilibriumModelSparse`, the **sparse** model, assembles the same system stored in sparse format and solves it with a sparse solver. Identical results, but it scales to large structures where a dense matrix would not fit.

Sparse is the default, and the one to keep for big unconstrained form-finding.
The one time it steps aside is when you add constraints: constrained form-finding runs on the dense model, and passing constraints with `sparse=True` quietly switches you to dense (with a printed heads-up).

### The equilibrium state

It is tempting to think of the model's output as "the shape," but an `EquilibriumState` is more than geometry.
It is the complete configuration of the pin-jointed bar system at equilibrium: where every node sits *and* how force flows through the whole assembly, the two halves of a static answer.
Give a bar network force densities and loads and the model returns not just the coordinates the structure settles into, but the axial force it carries, the reactions its supports must supply, and the residual at every free node that says whether equilibrium was actually reached.

The state is a named tuple of arrays:

| Field | Shape | What it holds |
| --- | --- | --- |
| `xyz` | `(nodes, 3)` | the equilibrium coordinates of every node |
| `residuals` | `(nodes, 3)` | the residual force at each node, zero at free nodes in equilibrium, the reaction at supports |
| `lengths` | `(edges, 1)` | the length of each edge |
| `forces` | `(edges, 1)` | the axial force in each edge, signed like its force density |
| `loads` | `(nodes, 3)` | the load resolved onto each node |
| `vectors` | `(edges, 3)` | the edge vectors, pointing from tail node to head node |

The **geometry** lives in `xyz`, `lengths`, and `vectors`; the **force state** lives in `forces`, `residuals`, and `loads`.
The residuals do double duty: they are your equilibrium check (a free node with a non-zero residual has not settled) and your reactions (a support's residual *is* the force it feeds back into the structure).

This is the object `fdm` unpacks to update the datastructure it returns.
It is also, verbatim, the `eq_state` a goal's `prediction(eq_state, index)` slices into: when you write a custom goal, you are reading one row out of these arrays, whether you care about a coordinate, a length, or a force.
So the numerical core and the [goals](goals.md) guide meet right here.

For the full signatures and every method, see the [equilibrium API reference](../api/jax_fdm.equilibrium.md).

## Where to next

- To steer form-finding toward a design intent, minimizing, bounding, and targeting quantities, read [constrained form-finding](constrained_form_finding.md).
- To teach JAX FDM a quantity it does not measure yet, write a [custom goal](custom_goals.md) or [custom constraint](custom_constraints.md).
