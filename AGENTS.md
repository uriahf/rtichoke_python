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
