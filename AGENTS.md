# rtichoke Agent Information

This document provides guidance for AI agents working on the `rtichoke` repository.

## Development Environment

To set up the development environment, follow these steps:

1. **Install `uv`**.
2. **Create a virtual environment** with `uv venv`.
3. **Install development dependencies** with:

    ```bash
    uv sync --dev
    ```

## Running Tests

Run the test suite with:

```bash
uv run pytest
```

## Coding Conventions

### Functional Programming

Strive to use a functional programming style as much as possible. Avoid side effects and mutable state where practical.

### Docstrings

All exported functions must have NumPy-style docstrings. Great Docs parses these docstrings to generate the API reference, so parameter, return-value, and usage documentation should remain accurate and user-facing.

Example:

```python
def my_function(param1, param2):
    """Summary of the function's purpose.

    Parameters
    ----------
    param1 : int
        Description of the first parameter.
    param2 : str
        Description of the second parameter.

    Returns
    -------
    bool
        Description of the return value.
    """
    return True
```

## Pre-commit Hooks

This repository uses pre-commit hooks for code quality and consistency, including `ruff-check`, `ruff-format`, and `uv-lock`.

Run them manually with:

```bash
pre-commit run --all-files
```

## Documentation

Documentation is built with Great Docs and Quarto.

- Great Docs configuration: `great-docs.yml`
- Narrative guides: `user_guide/`
- Documentation dependencies: the `docs` dependency group in `pyproject.toml`
- Local build: `uv sync --group docs` followed by `uv run great-docs build`
- Pull requests receive a rendered preview under the repository's GitHub Pages site.
- Merges to `main` publish the production documentation automatically.

Great Docs requires Python 3.11 or newer for documentation builds. This does not change the package's Python >=3.9 runtime support.

## Type Checking

This project uses `ty` for type checking. Run:

```bash
uv run ty check src tests
```

## PR completion protocol

Do not consider an implementation task complete merely because code has been pushed or a pull request has been opened.

After creating or updating a pull request:

1. Inspect all required GitHub Actions checks for the current PR head.
2. If checks are still running, re-check them while the session is active rather than handing the PR back to the user for manual monitoring.
3. If a required check fails, inspect the failing job and logs and determine whether the failure is caused by the PR.
4. If the fix is within the stated task scope, make the fix, push it, and inspect CI again.
5. Repeat the diagnose/fix/re-check loop until all required checks pass or a genuine blocker requires user input.

Escalate to the user only when resolving the failure would require one or more of the following:

- changing frozen statistical semantics, contracts, or architecture;
- broadening the agreed task scope;
- weakening or removing a meaningful test or quality gate;
- changing a public API or backward-compatibility promise beyond the task;
- making a product or technical decision with multiple legitimate choices;
- resolving an external service, permissions, infrastructure, or credential problem that the agent cannot fix safely.

Routine failures such as lint errors, formatting errors, test regressions caused by the PR, snapshots/fixtures that legitimately need updating, packaging errors, documentation-build errors, and similar mechanical issues should be fixed without asking the user to manually inspect GitHub Actions.

The final handoff should include:

- pull request link;
- final PR head commit;
- tests/checks run locally when applicable;
- final GitHub Actions status;
- any remaining caveats or blockers.

Do not ask the user to manually check whether CI passed.
