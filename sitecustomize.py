# Lightweight import-time hook to disable success-state saving without touching generated files.
# Activate by setting environment variable GENHRL_DISABLE_SUCCESS_SAVING=1.

import os
import sys
import importlib.abc
import importlib.machinery
import importlib.util
from types import ModuleType


def _should_disable() -> bool:
    return os.environ.get("GENHRL_DISABLE_SUCCESS_SAVING", "0") == "1"


class _SuccessSavePatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped_loader: importlib.abc.Loader):
        self._fullname = fullname
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        # Execute real module first
        self._wrapped_loader.exec_module(module)
        # Then patch if needed
        try:
            if self._fullname.endswith("SuccessTerminationCfg"):
                # Replace any imported save_success_state symbol in this module's namespace
                if hasattr(module, "save_success_state"):
                    def _noop(*args, **kwargs):
                        return None
                    module.save_success_state = _noop  # type: ignore[attr-defined]
            elif self._fullname.endswith("base_success"):
                # Also patch base_success utilities as a safety net
                if hasattr(module, "save_success_state"):
                    def _noop(*args, **kwargs):
                        return None
                    module.save_success_state = _noop  # type: ignore[attr-defined]
                if hasattr(module, "save_states_to_disk"):
                    def _noop2(*args, **kwargs):
                        return None
                    module.save_states_to_disk = _noop2  # type: ignore[attr-defined]
        except Exception:
            # Never break the import if patching fails
            pass


class _SuccessSavePatchFinder(importlib.abc.MetaPathFinder):
    TARGET_SUFFIXES = ("SuccessTerminationCfg", "base_success")

    def find_spec(self, fullname: str, path, target=None):
        if not any(fullname.endswith(suf) for suf in self.TARGET_SUFFIXES):
            return None
        # Delegate real finding to PathFinder to avoid recursion
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        # Wrap the original loader
        spec.loader = _SuccessSavePatchLoader(fullname, spec.loader)
        return spec


# Install only when requested
if _should_disable():
    # Insert at the front so we wrap before default path finder
    sys.meta_path.insert(0, _SuccessSavePatchFinder())
