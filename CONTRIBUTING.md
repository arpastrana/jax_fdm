# Contributing

Contributions are welcome and very much appreciated!

## Code contributions

We accept code contributions through pull requests.
In short, this is how that works.

1. Fork [the repository](https://github.com/arpastrana/jax_fdm) and clone the fork.
2. Create a virtual environment using your tool of choice (e.g. `virtualenv`, `conda`, etc).
3. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Install the pre-commit hooks (they run `ruff` and basic file checks on each commit):

   ```bash
   pre-commit install
   ```

5. Make sure all tests pass:

   ```bash
   invoke test
   ```

6. Start making your changes to the **main** branch (or branch off of it).
7. Make sure all tests still pass:

   ```bash
   invoke test
   ```

8. Add yourself to the *Contributors* section of `AUTHORS.md`.
9. Commit your changes and push your branch to GitHub.
10. Create a [pull request](https://help.github.com/articles/about-pull-requests/) through the GitHub website.

During development, use [pyinvoke](http://docs.pyinvoke.org/) tasks on the
command line to ease recurring operations:

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
