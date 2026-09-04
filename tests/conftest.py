"""
Shared test configuration.

The pyproject.toml [tool.pytest.ini_options] sets pythonpath = ["src"]
which makes the project root importable without sys.path hacks.
This conftest exists to ensure pytest discovers this directory.
"""
