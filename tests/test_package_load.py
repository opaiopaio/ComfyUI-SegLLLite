"""Regression: the package must load even though ComfyUI already has a top-level
module called `nodes` (its own nodes.py) in sys.modules.

This is exactly the bug that made every node silently disappear:
    from nodes.lllite_segs import ...   ->  resolved to ComfyUI's nodes.py

Run: <comfyui python> tests/test_package_load.py
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- stubs ComfyUI-like environment --------------------------------------- #
if "folder_paths" not in sys.modules:
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda kind: []
    fp.get_full_path = lambda kind, name: None
    fp.get_folder_paths = lambda kind: []
    fp.models_dir = "/fake"
    sys.modules["folder_paths"] = fp

# ComfyUI's own top-level module with the SAME name as our old subpackage.
fake_nodes = types.ModuleType("nodes")
fake_nodes.NODE_CLASS_MAPPINGS = {}
sys.modules["nodes"] = fake_nodes

spec = importlib.util.spec_from_file_location(
    "ComfyUI_LLLiteSEGS",
    os.path.join(ROOT, "__init__.py"),
    submodule_search_locations=[ROOT],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_LLLiteSEGS"] = mod
spec.loader.exec_module(mod)

maps = mod.NODE_CLASS_MAPPINGS
ok = True
expected = [
    "LLLiteApplySEGS",
    "LLLiteClearSEGS",
    "LLLiteHintPreviewSEGS",
]
for k in expected:
    good = k in maps
    ok = ok and good
    print(f"[{'OK ' if good else 'FAIL'}] registered {k}")

print(f"[{'OK ' if len(maps) >= 3 else 'FAIL'}] total nodes = {len(maps)}")
print("RESULT:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)