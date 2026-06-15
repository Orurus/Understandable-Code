"""Custom module loader for UnderstandableCode .und files and Python modules."""

import importlib
import importlib.util
import os
import sys
import types


MODULE_ALIASES = {
    "loader": "undml.load.module_loader",
    "tensor_matrix_library": "undml.tensor.tensor_matrix_operation",
    "advanced_ml_library": "undml.adv.advanced_ml_operation",
    "tensor_matrix_operation": "undml.tensor.tensor_matrix_operation",
    "advanced_ml_operation": "undml.adv.advanced_ml_operation",
    "forge_helpers": "undml.tensorforge.forge_helpers",
}


def _candidate_paths():
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.dirname(package_dir)
    return [
        src_dir,
        os.path.join(src_dir, "modules"),
        os.path.join(src_dir, "undml"),
        os.path.join(src_dir, "undml", "load"),
        os.path.join(src_dir, "undml", "tensor"),
        os.path.join(src_dir, "undml", "adv"),
        os.path.join(src_dir, "undml", "tensorforge"),
        package_dir,
        os.getcwd(),
    ]


def _normalize_module_path(module_name: str):
    """Normalize bare names and slash-delimited file paths into a real filesystem path."""
    if os.path.exists(module_name):
        return module_name

    candidate = module_name
    if candidate.startswith('/'):
        candidate = candidate.lstrip('/')

    if candidate.endswith('.und'):
        for base_dir in _candidate_paths():
            full_path = os.path.normpath(os.path.join(base_dir, candidate))
            if os.path.exists(full_path):
                return full_path

    return None


def load_simp_module(module_name: str):
    """Load a UnderstandableCode module from a .und file or Python file."""
    # First try regular Python import for modules such as tensor_matrix_library.
    try:
        return importlib.import_module(module_name)
    except Exception:
        pass

    alias_name = MODULE_ALIASES.get(module_name)
    if alias_name:
        try:
            return importlib.import_module(alias_name)
        except Exception:
            pass

    normalized_path = _normalize_module_path(module_name)
    if normalized_path:
        from core.codegen import compile_simp_to_py

        with open(normalized_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        python_code = compile_simp_to_py(source, normalized_path)

        module = types.ModuleType(os.path.splitext(os.path.basename(normalized_path))[0])
        module.__file__ = normalized_path
        exec(compile(python_code, normalized_path, "exec"), module.__dict__)
        sys.modules[module_name] = module
        return module

    for base_dir in _candidate_paths():
        simp_path = os.path.join(base_dir, f"{module_name}.und")
        if os.path.exists(simp_path):
            from core.codegen import compile_simp_to_py

            with open(simp_path, "r", encoding="utf-8") as handle:
                source = handle.read()
            python_code = compile_simp_to_py(source, simp_path)

            module = types.ModuleType(module_name)
            module.__file__ = simp_path
            exec(compile(python_code, simp_path, "exec"), module.__dict__)
            sys.modules[module_name] = module
            return module

    raise ImportError(f"Could not import UnderstandableCode module '{module_name}'")
