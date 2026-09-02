# Contributing

Contributions are welcome and very much appreciated!

## Code contributions

We accept code contributions through pull requests.
In short, this is how that works.

1. Fork [the repository](https://github.com/arpastrana/jax_fdm) and clone the fork.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it.
3. Create the development environment. This installs a supported Python, the package in editable mode, the visualization extra and every development tool, pinned by `uv.lock`:

   ```bash
   uv sync --all-groups --extra viz
   ```

   Prefix commands with `uv run` to execute them inside that environment, no activation needed.
   If you prefer `pip`, create a virtual environment on Python 3.11 to 3.13 and run `pip install -e ".[viz]" --group dev --group docs --group typecheck` (pip 25.1 or newer).

4. Install the pre-commit hooks (they run `ruff` and basic file checks on each commit):

   ```bash
   uv run pre-commit install
   ```

5. Make sure all tests pass:

   ```bash
   uv run invoke test
   ```

6. Start making your changes to the **main** branch (or branch off of it).
7. Make sure all tests still pass:

   ```bash
   uv run invoke test
   ```

8. Add yourself to the *Contributors* section of `AUTHORS.md`.
9. Commit your changes and push your branch to GitHub.
10. Create a [pull request](https://help.github.com/articles/about-pull-requests/) through the GitHub website.

During development, use [pyinvoke](http://docs.pyinvoke.org/) tasks on the
command line to ease recurring operations (prefix each with `uv run`):

* `invoke clean`: Clean all generated artifacts.
* `invoke lint`: Check the coding style with ruff.
* `invoke docs`: Build the documentation site with mkdocs. Pass `--serve` to preview it locally with live reload.
* `invoke test`: Run all tests and checks in one swift command.
* `invoke`: Show available tasks.

## Bug reports

When [reporting a bug](https://github.com/arpastrana/jax_fdm/issues) please include:

* Operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Feature requests

When [proposing a new feature](https://github.com/arpastrana/jax_fdm/issues) please include:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
