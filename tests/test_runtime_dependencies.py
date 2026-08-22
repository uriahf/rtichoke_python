from pathlib import Path


def test_pandas_is_not_a_direct_runtime_dependency():
    pyproject = Path("pyproject.toml").read_text()
    project_dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert '"pandas' not in project_dependencies
