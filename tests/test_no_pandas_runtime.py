import builtins
import importlib
import sys


def test_core_rtichoke_imports_do_not_require_pandas(monkeypatch):
    """Core runtime modules must import when pandas is unavailable."""
    real_import = builtins.__import__

    def import_without_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ModuleNotFoundError("pandas intentionally unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pandas)

    for module_name in list(sys.modules):
        if module_name == "rtichoke.processing.adjustments":
            sys.modules.pop(module_name, None)

    importlib.import_module("rtichoke.processing.adjustments")
