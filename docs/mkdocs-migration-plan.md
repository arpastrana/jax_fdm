# MkDocs documentation plan for jax_fdm

Handover document. Prepared 2026-07-10 on branch `docs` from a four-agent analysis:
(1) public-API docstring/annotation audit, (2) current docs/CI inventory,
(3) equinox docs teardown, (4) COMPAS package template docs teardown.

Goal: replace the dead Sphinx skeleton in `docs/` with an auto-generating
**mkdocs + mkdocs-material + mkdocstrings** site, deployed to GitHub Pages via CI.

---

## 1. Current state (what exists today)

- `docs/` is a **Sphinx skeleton with no content**: `conf.py` (imports the
  discontinued `sphinx_compas_theme` v1, stale `release = "0.10.0"`, broken URLs with
  a missing org slug), 8 `.rst` files of which `installation.rst` and `tutorial.rst`
  are title-only stubs, `api/jax_fdm.rst` is a single `automodule` line. No Makefile.
  There is essentially **no prose to port** — the README (218 lines, rich) is the real
  landing-page content.
- `.github/workflows/docs.yml` exists but its entire `jobs:` block is **commented
  out** (dead `compas-dev/compas-actions.docs@v2.2.1` reference).
- A stale `origin/gh-pages` branch holds a Sphinx build from **2023-05-31**. No CNAME.
  Pages has been dormant ~3 years. Repo: `arpastrana/jax_fdm` → target URL
  `https://arpastrana.github.io/jax_fdm/`.
- `pyproject.toml` has `viz` / `ipopt` / `dev` extras but **no `docs` group**; no
  sphinx or mkdocs dependency is declared anywhere.
- `CONTRIBUTING.md` mentions `invoke docs`, but `tasks.py` has no `docs` task — the
  old workflow is already broken. README has no docs badge; pyproject has no
  `Documentation` URL.
- Other assets: `notebooks/` (3 Colab notebooks), `examples/` (14 scripts, only ~5
  with module docstrings, mostly generic duplicates), `CHANGELOG.md` (actively
  maintained, enforced by `pr-checks.yml`).

**Implication:** this is not a migration — the docs are currently unbuildable and
undeployed. Adopting mkdocs is effectively greenfield; the whole `docs/` folder can
be replaced wholesale.

## 2. API surface audit (blast radius of the docstring work)

Source: `src/jax_fdm/`, 121 Python files. Public (non-underscore) API only:

| Subpackage | Public funcs | Public classes | Public methods | Docstring gaps |
|---|---|---|---|---|
| equilibrium | 61 | 15 | 44 | 5 classes |
| geometry | 27 | 0 | 0 | none |
| visualization | 21 | 29 | 93 | 4 classes, **65 methods** |
| goals | 7 | 43 | 81 | 2 classes, 9 methods |
| parameters | 3 | 41 | 45 | 2 classes, 4 methods |
| optimization | 1 | 22 | 27 | 1 method |
| constraints | 0 | 17 | 33 | 5 methods |
| datastructures | 0 | 3 | 56 | none |
| losses | 0 | 12 | 20 | 2 classes, 12 methods |
| **Total** | **120** | **182** | **399** | **15 classes + 96 methods** |

≈ **701 public documentable objects**; docstring *presence* is 84% (590/701), but:

- Most existing docstrings are **one-line freeform summaries** with no
  Parameters/Returns sections. Real numpy-style docstrings exist in ~21 files,
  almost all in `visualization/` (e.g. `visualization/style.py`, `backends.py`,
  `shapes.py`). So beyond the ~111 missing docstrings, **most of the ~590 existing
  ones need expansion** to render as useful parameter tables.
- **Type annotations: essentially zero.** 0 of ~743 function/method signatures have
  any argument or return annotation. The only annotations in the package are 33
  equinox dataclass field annotations (`jax.Array`, `np.ndarray`) in
  `equilibrium/structures/*.py` and `states.py`. Signature annotations would be a
  from-scratch effort across ~700 objects.

### Autodoc hazards found in the codebase

1. **Star imports + dynamic `__all__` everywhere.** 34 `from .X import *`
   statements; nearly every `__init__.py` computes
   `__all__ = [n for n in dir() if not n.startswith('_')]` at import time.
   → Document objects **by canonical module path** in `:::` stubs (or with explicit
   `members:` lists), not by slurping packages. Consider replacing dynamic `__all__`
   with static lists over time.
2. **Conditional viz imports.** `visualization/{plotters,viewers,notebooks}/__init__.py`
   star-import inside availability guards. A docs build without the `viz` extras
   silently drops those symbols. → Docs CI must `pip install ".[viz,docs]"`, or the
   API stubs must target the leaf modules directly. compas_viewer pulls
   PySide6/OpenGL onto a headless runner — **test `mkdocs build` locally against the
   visualization stubs before wiring CI**; fall back to documenting leaf modules
   statically if imports fail headless.
3. **`custom_vjp`-wrapped solver functions** at module scope
   (`equilibrium/solvers/nonlinear.py:11`, `fixed_point.py:138`) may not expose
   normal signatures to runtime inspection. → Prefer griffe's default **static
   analysis** (do not set `force_inspection: true`); verify these two render.
4. **equinox `Module` classes** (`Graph`, `GraphSparse`, `Mesh(Sparse)`,
   `EquilibriumStructure` + `IndexingMixins`) are dataclass-like; check generated
   `__init__`/field rendering, use `merge_init_into_class: true`.

## 3. What to copy from equinox and the COMPAS template

Both reference sites use mkdocs-material + mkdocstrings with hand-written `:::`
stub pages — **neither auto-generates API pages** (no gen-files/literate-nav).
That is the model to follow.

### From equinox (docs.kidger.site/equinox)

**Copy:**
- `strict: true` in mkdocs.yml — warnings (broken cross-refs, malformed docstrings)
  fail the build. Combined with a build-on-PR check this is free regression
  protection and will drive the docstring cleanup.
- **Curated, thematic API pages**: hand-written markdown with `:::` directives,
  prose between groups, `---` separators, explicit `members:` whitelists. This is
  why equinox docs feel designed rather than dumped.
- **Exact version pins** for the docs dependency group — the
  mkdocs/mkdocstrings/griffe trio breaks across versions constantly.
- Custom CSS ideas: `div.doc-contents` left-border indent and boxed
  `h4.doc-heading` signature headings (two rules carry most of the "equinox look").
- Search `separator` regex so `EdgeLengthGoal` matches `edge length` and dotted
  paths split on `.`.
- `pymdownx.arithmatex` + MathJax (form-finding docs want equilibrium equations);
  copy `docs/_static/mathjax.js` verbatim.
- Nav shape: Getting started → a single **"All of jax_fdm"** conceptual tour page
  (FDM → goals → loss → optimizer pipeline in one narrative) → Examples → API →
  FAQ/Citation.
- Notebooks committed **with outputs**, listed directly in nav, never executed at
  build time.

**Skip (equinox-specific):**
- `hippogriffe` and `mkdocs-ipynb` (Kidger's personal tooling for
  jaxtyping/overload-heavy APIs and API-crossrefs-in-notebooks; use stock
  mkdocstrings + `mkdocs-jupyter` instead).
- `force_inspection: true`, `custom_dir` theme partial overrides,
  `include_exclude_files`/`.htaccess` (artifacts of his personal Apache hosting —
  equinox doesn't even deploy from CI; jax_fdm should).
- `toc.integrate` (hurts long numpy-style API pages with parameter tables; try
  without it first).
- Do **not** adopt equinox's markdown-bullet docstring house style. jax_fdm keeps
  numpy style; griffe parses it natively into real parameter tables
  (`docstring_style: numpy`) — more structure than equinox gets.

### From the COMPAS package template (compas-dev/compas_package_template)

Confirmed: the template is mkdocs-material + mkdocstrings + mike, deployed to
`gh-pages`. COMPAS core and compas_viewer are still Sphinx; **compas_notebook**
(compas-dev.github.io/compas_notebook) is the live example of the mkdocs stack and
the better model in several places.

**Copy:**
- The **mkdocstrings handler options block nearly verbatim** (see §5) — tuned for
  numpy style: `merge_init_into_class`, `separate_signature`, `filters: public`,
  `signature_crossrefs`, etc.
- **Sphinx inventories for cross-linking**: `https://compas.dev/compas/latest/objects.inv`
  makes `compas.datastructures.Mesh` references live links into COMPAS core docs.
  Add `https://docs.python.org/3/objects.inv`, `https://numpy.org/doc/stable/objects.inv`,
  and `https://docs.jax.dev/en/latest/objects.inv`.
- **Per-subpackage one-line `:::` stub pages** (compas_notebook's `docs/reference/`
  pattern): `docs/api/jax_fdm.goals.md` containing `# ::: jax_fdm.goals` etc.
  ~10 tiny files, zero machinery. (But see hazard §2.1 — where dynamic `__all__`
  pulls in noise, add explicit `members:` lists or target leaf modules.)
- **COMPAS-blue branding**: the template's entire theme customization is one CSS
  rule (`--md-primary-fg-color: #0092d2`) — instant "COMPAS-family" visual identity.
  Take the template's light/dark toggle palette.
- The `markdown_extensions` block (admonition, attr_list, pymdownx highlight/
  inlinehilite/snippets/superfences, toc permalink).
- `[tool.ruff.lint.pydocstyle] convention = "numpy"` in pyproject so docstrings stay
  parseable by construction.
- **compas_notebook's workflow shape** (not the template's): ~25 transparent lines,
  checkout → setup-python → mkdocs-material cache → `pip install ".[docs]"` →
  `mkdocs gh-deploy --force` on push to main.

**Skip (template bloat / wrong fit):**
- `compas-actions.docs@v5` composite action: in mkdocs mode it only deploys on
  version tags (not on main), hard-depends on `compas_invocations2` invoke tasks
  jax_fdm doesn't use, and installs graphviz/xvfb/latex. jax_fdm already abandoned
  an older version of this action.
- **mike versioning**: configured by the template, unused even by compas_notebook.
  Drop `extra.version`; add mike later if per-release docs ever matter.
- Three-quarters of the template's plugin list (`mkdocs-coverage`, `mkdocs-llmstxt`,
  `mkdocs-minify`, `mkdocs-redirects`, `mermaid2`, `markdown-exec`, …) — installed
  by the template, enabled by none of its config. Start minimal.
- `compas_invocations2`/`tasks.py` rework and Grasshopper build steps.
- The template's single `# ::: package` whole-package page — documents only
  top-level re-exports; jax_fdm's API spans subpackages, use per-subpackage stubs.

### A note on equinox's docstring style (and why not to adopt it)

Equinox does **not** use numpy/Google style. It uses a markdown house format that
griffe never parses, rendered verbatim:

```python
"""**Arguments:**

- `in_features`: The input size. The input should be a vector of shape `(in_features,)`.

**Returns:**

A JAX array of shape `(out_features,)`.
"""
```

Its real advantages: full markdown freedom inside docstrings (admonitions, code
fences, `[equinox.Module][]` cross-links anywhere), no section grammar to satisfy,
and — by having no type slot in the docstring — it enforces types-live-in-signatures
by construction, pairing with jaxtyping to keep docstrings very short.

Rejected for jax_fdm because: it loses structured parameter tables and griffe's
section validation (which under `strict: true` acts as a free docstring linter);
COMPAS is numpy-style throughout and jax_fdm's existing real docstrings already are
too; ruff can enforce `convention = "numpy"`; and the one capability numpy lacks —
inline markdown mid-docstring — mostly still works, since mkdocstrings renders
markdown inside numpy description fields. **Keep numpy style, but steal the
underlying principle** — see the policy below.

## 4. Docstring & annotation policy (decided)

Settle this *before* writing the ~590 expanded docstrings; retrofitting the policy
later means touching every docstring twice.

1. **Types live in signatures, not in docstrings.** Annotate public signatures and
   write numpy Parameters entries *without* the type token (`edges :` instead of
   `edges : tuple of int`). mkdocstrings fills the parameter table's type column
   from the annotation automatically. Single source of truth, checkable with
   pyright/mypy (a docstring type line is never checked by anything), no drift.
   Note the savings are in non-duplication and checkability more than volume —
   every parameter still needs its description line.
2. **jaxtyping in the array-facing core.** Plain `jax.Array` undersells this
   domain — what users need is *shape* ("array of shape `(n_edges,)` of force
   densities"), and without it shapes creep back into the prose, recreating the
   duplication. Use `Float[Array, "n_edges"]`-style annotations in `equilibrium`,
   `geometry`, and goal/constraint prediction methods; plain annotations
   (`float`, COMPAS types, `jax.Array`) elsewhere. jax_fdm already depends on
   equinox, so jaxtyping is a natural, light addition — but it is a real API-design
   commitment (scalar-vs-array unions, PyTrees, COMPAS-object inputs all need
   decisions). If the shape decisions stall progress, fall back to plain
   annotations there too and revisit.
3. **One combined pass per subpackage, not two global passes.** Doing all
   annotations first and all docstrings second touches every file twice and holds
   the whole API in your head twice. Instead, annotate + write docstrings together,
   one subpackage at a time — the type is decided at the exact moment the
   parameter's description is written, and each subpackage lands as one reviewable
   PR (which also suits the CHANGELOG-per-PR gate).

## 5. Proposed architecture

```
mkdocs.yml                      # material theme, strict: true, COMPAS blue
docs/
  index.md                      # landing page, distilled from README
  installation.md               # incl. viz/ipopt extras, Windows note
  all-of-jax-fdm.md             # single-page conceptual tour (equinox pattern)
  examples/
    index.md                    # gallery: pre-rendered images + snippets/links
    arch.ipynb ...              # (optional) the 3 notebooks via mkdocs-jupyter
  api/
    index.md                    # API overview / how the package is organized
    jax_fdm.datastructures.md   # one-line ::: stubs, one per subpackage
    jax_fdm.equilibrium.md
    jax_fdm.goals.md            # possibly split goals/ by element type later
    jax_fdm.constraints.md
    jax_fdm.losses.md
    jax_fdm.optimization.md
    jax_fdm.parameters.md
    jax_fdm.geometry.md
    jax_fdm.visualization.md    # the fragile one — verify headless build
  citation.md                   # 3 BibTeX entries from README
  license.md
  assets/css/custom.css         # COMPAS blue + equinox doc-heading/doc-contents rules
  assets/js/mathjax.js
```

Delete: `docs/conf.py`, all `docs/*.rst`, `docs/api/jax_fdm.rst`, PLACEHOLDER dirs.

Examples strategy: the viz output is a GUI viewer window that cannot render
headless, so **do not build a script-execution gallery** (no mkdocs-gallery/
papermill). Use hand-written example pages with pre-rendered images (README already
has some) + `pymdownx.snippets` embedding of the script source, plus links to the
Colab notebooks. Adding module docstrings to the ~9 examples that lack them is
nice-to-have, not blocking.

### mkdocs.yml starting point

```yaml
site_name: JAX FDM
site_url: https://arpastrana.github.io/jax_fdm/
repo_url: https://github.com/arpastrana/jax_fdm
strict: true

theme:
  name: material
  features: [navigation.sections, navigation.top, content.code.copy,
             search.highlight, search.suggest, toc.follow]
  palette:
    - scheme: default
      toggle: {icon: material/weather-night, name: Dark mode}
    - scheme: slate
      toggle: {icon: material/weather-sunny, name: Light mode}

extra_css: [assets/css/custom.css]
extra_javascript:
  - assets/js/mathjax.js
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

markdown_extensions:
  - admonition
  - attr_list
  - footnotes
  - pymdownx.arithmatex: {generic: true}
  - pymdownx.details
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.snippets: {base_path: [docs, examples]}
  - pymdownx.superfences
  - toc: {permalink: "¤"}

plugins:
  - search:
      separator: '[\s\-,:!=\[\]()"/]+|(?!\b)(?=[A-Z][a-z])|\.(?!\d)'
  - mkdocs-jupyter          # only if notebooks go in the site
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src]
          inventories:
            - https://docs.python.org/3/objects.inv
            - https://compas.dev/compas/latest/objects.inv
            - https://numpy.org/doc/stable/objects.inv
            - https://docs.jax.dev/en/latest/objects.inv
          options:
            docstring_style: numpy
            docstring_options: {ignore_init_summary: true, trim_doctest_flags: true}
            docstring_section_style: table
            filters: public
            heading_level: 2
            members_order: source
            merge_init_into_class: true
            separate_signature: true
            show_bases: false
            show_root_heading: true
            show_signature_annotations: true
            show_source: false
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            signature_crossrefs: true
```

### Dependencies (pyproject, exact pins per equinox practice)

```toml
[project.optional-dependencies]
docs = [
  "mkdocs==1.6.*",
  "mkdocs-material==9.6.*",
  "mkdocstrings[python]==0.28.*",
  "mkdocs-jupyter==0.25.*",   # only if notebooks included
]
```

(Pin exact patch versions at implementation time; verify griffe compatibility.)

### CI

Replace the commented-out `docs.yml` with two jobs:

1. **PR check** (`pull_request` to main): `pip install ".[viz,docs]"` +
   `mkdocs build --strict`. Build only — never deploy from PRs. Honor the existing
   `docs-only` label convention if useful.
2. **Deploy** (`push` to main): same install + `mkdocs gh-deploy --force`
   (overwrites the stale 2023 `gh-pages` content; nothing there to preserve).
   Set `JAX_PLATFORMS=cpu`. One-time manual step: repo Settings → Pages → deploy
   from `gh-pages` branch (verify it's still configured).

Follow-ups in the same pass: add `Documentation` URL to `[project.urls]`, docs badge
+ link in README's "Documentation" section, fix broken `github.com//jax_fdm` links
in CONTRIBUTING.md, delete the `invoke docs` mention or add a thin
`mkdocs build/serve` invoke task.

## 6. Blast radius

**Mechanical/infra changes (small, contained):**
- `docs/` wholesale replacement (~11 files deleted, ~18 created). No other code
  reads `docs/`.
- `.github/workflows/docs.yml` rewritten (currently dead — zero behavioral risk).
- `pyproject.toml`: add `docs` extra + `Documentation` URL + ruff pydocstyle
  convention. No runtime impact.
- `gh-pages` branch force-overwritten (stale 2023 build, nothing to preserve).
- README/CONTRIBUTING link touch-ups.

**Source-code changes (the real blast radius):**
- Docstrings touch **up to ~700 objects across ~120 files** — but docstring-only
  edits are behaviorally inert *with one caveat*: pytest runs with
  `--doctest-modules` (see `[tool.pytest.ini_options]`), so **any `>>>` examples
  added to docstrings become executed tests**. That's a feature (validated
  examples) but means docstring PRs can fail CI legitimately.
- Type annotations touch every signature (see §4 policy — done in the same pass
  as docstrings). They are also inert at runtime (Python doesn't evaluate them at
  call time; jax doesn't introspect them) but are a large review surface and need
  jax-typing care (`jax.Array` vs `ArrayLike` vs jaxtyping shapes vs COMPAS types
  vs PyTrees). jaxtyping does add runtime import weight and, if runtime
  checking is ever enabled, behavior — keep it annotation-only.
- Optional structural hardening (static `__all__` lists) changes import-time
  behavior slightly — do carefully, subpackage by subpackage, with the test suite.
- Every docstring PR must carry a CHANGELOG entry (`pr-checks.yml` enforces it) —
  batch the work into a few large PRs, not many small ones.

**No changes needed to**: runtime logic, tests (except doctests added on purpose),
examples, release workflow, packaging backend.

## 7. Effort estimate

Assume focused work, agent-assisted where mechanical.

| Phase | Work | Estimate |
|---|---|---|
| **1. Site skeleton** | mkdocs.yml, theme/CSS/mathjax, delete rst, index/install/citation/license pages from README, `docs` extra | ~half a day |
| **2. API stubs + hazard fixes** | 10 `:::` stub pages, verify equinox-Module + custom_vjp + viz rendering headless, member whitelists where dynamic `__all__` leaks noise | ~half–1 day |
| **3. CI + deploy** | build-check + gh-deploy workflows, Pages settings, badge/URLs, kill stale gh-pages | ~half a day |
| **4. Docstring completion** | write the ~111 missing docstrings (65 are visualization methods, many settable-property boilerplate) | 1–2 days |
| **5. Docstrings + annotations, combined pass** | per §4 policy: annotate signatures AND upgrade one-liners to typeless numpy sections, one subpackage per PR. Prioritize the user-facing core: `goals`, `constraints`, `losses`, `equilibrium.fdm`, `optimization`, `datastructures`, `parameters` (~250–300 objects). `strict: true` will surface malformed sections | 5–8 days (3–5 for docstrings alone; +2–3 for annotations, mostly the jaxtyping shape decisions in the array core) |
| **6. Narrative content** | `all-of-jax-fdm.md` tour, examples pages with images, notebook integration | 1–2 days |
| **7. Annotation long tail** (optional) | signatures outside the phase-5 core (visualization internals, private-ish helpers); mypy/pyright config if type-checking is wanted | ~1 week, incremental; **doesn't gate the site** |

**Bottom line:** a live, styled, auto-generated API site (phases 1–3) is **~2 days**.
A genuinely good site with complete docstrings, annotated core signatures, and
narrative pages (phases 4–6) is **another ~2 weeks**. The remaining annotation long
tail is incremental follow-up work the site does not depend on.

Recommended order: ship phases 1–3 first (the site immediately renders the 84% of
docstrings that exist), then land docstring work in a few large per-subpackage PRs
with the strict build as the quality gate.

## 8. Open decisions for the implementer

1. **Notebooks in the site or Colab-links only?** mkdocs-jupyter is one plugin +
   committed outputs; Colab links already work from README. Suggest: start with
   links, add mkdocs-jupyter in phase 6.
2. **`visualization` API pages**: if headless import of compas_viewer fails in CI,
   either install xvfb, document leaf modules statically, or drop viewer classes
   from the API reference (they're thin wrappers). Decide after a local
   `mkdocs build` test.
3. **Static `__all__` migration**: worthwhile hygiene but not required if stubs use
   explicit members. Defer unless mkdocstrings output is noisy.
4. **goals/ page granularity**: one page for all 43 goal classes vs. split by
   element (node/edge/mesh/network/vertex/face). Suggest split — mirrors the
   package layout and keeps pages scannable.
5. **Docstring one-liner policy**: expand *everything* or accept one-liners for
   trivial properties? Suggest: full numpy sections for public entry points
   (goals/constraints/losses/fdm/optimizers), one-liners acceptable for property
   getters/setters.
6. **jaxtyping scope** (§4.2): exactly which modules get shape annotations vs
   plain `jax.Array`, and how to annotate scalar-or-array unions and
   COMPAS-or-array inputs. Decide per-subpackage during the phase-5 pass; don't
   let it block — plain annotations are an acceptable fallback anywhere.
