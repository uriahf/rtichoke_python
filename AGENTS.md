# rtichoke Agent Information

This document provides guidance for AI agents working on the `rtichoke` repository.

## Development Environment

To set up the development environment, follow these steps:

1. **Install `uv`**: If you don't have `uv` installed, please follow the official installation instructions.
2. **Create a virtual environment**: Use `uv venv` to create a virtual environment.
3. **Install dependencies**: Install the project dependencies, including the `dev` dependencies, with the following command:

    ```bash
    uv pip install -e .[dev]
    ```

## Running Tests

The test suite is run using `pytest`. To run the tests, use the following command:

```bash
uv run pytest
```

## Coding Conventions

### Functional Programming

Strive to use a functional programming style as much as possible. Avoid side effects and mutable state where practical.

### Docstrings

All exported functions must have NumPy-style docstrings. This is to ensure that the documentation is clear, consistent, and can be easily parsed by tools like `quartodoc`.

Example of a NumPy-style docstring:

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
    # function body
    return True
```

## Pre-commit Hooks

This repository uses pre-commit hooks to ensure code quality and consistency. The following hooks are configured:

* **`ruff-check`**: A linter to check for common errors and style issues.
* **`ruff-format`**: A code formatter to ensure a consistent code style.
* **`uv-lock`**: A hook to keep the `uv.lock` file up to date.

Before committing, please ensure that the pre-commit hooks pass. You can run them manually on all files with `pre-commit run --all-files`.

## Documentation

The documentation for this project is built using `quartodoc`. The documentation is automatically built and deployed via GitHub Actions. There is no need to build the documentation manually.

## Type Checking

This project uses `ty` for type checking. To check for type errors, run the following command:

```bash
uv run ty check src tests
```
