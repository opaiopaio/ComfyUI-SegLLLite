"""Load the Anima-LLLite node class from its *package* (never `import nodes`).

Anima-LLLite layout::

    ComfyUI-Anima-LLLite/
      __init__.py          ->  from .nodes import NODE_CLASS_MAPPINGS, ...
      nodes.py             ->  from .control_net_lllite_anima import (...)
      control_net_lllite_anima.py

Because ``nodes.py`` uses **relative** imports, it can only be executed as a
submodule of its package. Loading it as a standalone file (the obvious
``spec_from_file_location("x", nodes.py)`` trick) fails with::

    ImportError: attempted relative import with no known parent package

So we resolve it in this order:

1. reuse the module ComfyUI already imported (``sys.modules``), which is the
   normal case — ComfyUI loads every custom node as a package;
2. otherwise import it by package name after adding ``custom_nodes`` to
   ``sys.path`` (``importlib`` accepts directory names with hyphens, unlike the
   ``import`` statement);
3. otherwise load ``__init__.py`` **as a package** with
   ``submodule_search_locations`` so relative imports still resolve.

The class we need (``AnimaLLLiteApply_sdscripts``) is a pure function of
``(model, lllite_name, image, strength, start_percent, end_percent,
preserve_wrapper, mask)`` with no ComfyUI execution-context dependency.
"""

import importlib
import importlib.util
import os
import sys
import threading

_LOG = "[SegLLLite]"

# Directory names seen in the wild for ComfyUI-Anima-LLLite.
CANDIDATE_DIRS = (
    "ComfyUI-Anima-LLLite",
    "comfyui-anima-lllite",
    "ComfyUI-Anima-LLLite-main",
    "comfyui_anima_lllite",
)

_CLASS_NAME = "AnimaLLLiteApply_sdscripts"

_lock = threading.Lock()
_module = None            # the package's `nodes` submodule
_base_override = None     # tests only


def set_base_dir(path):
    """Override the custom_nodes directory (tests / unusual installs)."""
    global _base_override, _module
    _base_override = path
    _module = None


def custom_nodes_dir() -> str:
    """``.../custom_nodes`` — this file lives at <pkg>/compat/anima_lllite.py."""
    if _base_override:
        return _base_override
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candidates(base):
    for dirname in CANDIDATE_DIRS:
        pkg_dir = os.path.join(base, dirname)
        if os.path.isdir(pkg_dir) and os.path.isfile(os.path.join(pkg_dir, "__init__.py")):
            yield dirname, pkg_dir


def find_nodes_file(base=None):
    """Path to the package's ``nodes.py`` (kept for diagnostics/back-compat)."""
    base = base or custom_nodes_dir()
    for dirname, pkg_dir in _candidates(base):
        path = os.path.join(pkg_dir, "nodes.py")
        if os.path.isfile(path):
            return path
    return None


def _from_sys_modules(names):
    for name in names:
        sub = sys.modules.get(name + ".nodes")
        if sub is not None and hasattr(sub, _CLASS_NAME):
            return sub
    return None


def _import_as_package(base):
    """Import the node package by directory name (hyphens are fine for importlib)."""
    if base not in sys.path:
        sys.path.insert(0, base)
    for dirname, _pkg_dir in _candidates(base):
        try:
            importlib.import_module(dirname)              # runs __init__.py
            sub = importlib.import_module(dirname + ".nodes")
            if hasattr(sub, _CLASS_NAME):
                return sub
        except Exception as e:  # keep trying the other candidate names
            print(f"{_LOG} import '{dirname}' failed: {e}")
    return None


def _load_by_path_as_package(pkg_dir, dirname):
    init = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        dirname, init, submodule_search_locations=[pkg_dir]
    )
    if spec is None or spec.loader is None:
        return None
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[dirname] = pkg
    spec.loader.exec_module(pkg)
    sub = sys.modules.get(dirname + ".nodes")
    if sub is None:
        sub = importlib.import_module(dirname + ".nodes")
    return sub if hasattr(sub, _CLASS_NAME) else None


def load_module(force: bool = False):
    """Return Anima-LLLite's ``nodes`` submodule; raises RuntimeError if absent."""
    global _module
    with _lock:
        if _module is not None and not force:
            return _module

        base = custom_nodes_dir()
        names = [d for d, _ in _candidates(base)]

        # 1) already imported by ComfyUI (normal case)
        sub = _from_sys_modules(names)
        if sub is not None:
            _module = sub
            return sub

        # 2) import as a package
        sub = _import_as_package(base)
        if sub is not None:
            _module = sub
            return sub

        # 3) load __init__.py as a package (relative imports still work)
        for dirname, pkg_dir in _candidates(base):
            try:
                sub = _load_by_path_as_package(pkg_dir, dirname)
            except Exception as e:
                print(f"{_LOG} load '{dirname}' by path failed: {e}")
                sub = None
            if sub is not None:
                _module = sub
                return sub

        from ..i18n import t  # relative: avoid clashing with a top-level i18n

        raise RuntimeError(t("err.no_anima_lllite"))


def get_apply_class():
    mod = load_module()
    cls = getattr(mod, _CLASS_NAME, None)
    if cls is None:
        raise RuntimeError(
            "Anima-LLLite is installed but does not expose "
            f"'{_CLASS_NAME}' (unsupported version)"
        )
    return cls


_apply_node = None


def get_apply_node():
    """A reusable instance; ``.apply()`` is stateless."""
    global _apply_node
    if _apply_node is None:
        _apply_node = get_apply_class()()
    return _apply_node


def _preserve_patcher_attrs(src, dst):
    """Copy ad-hoc attributes that ``ModelPatcher.clone()`` drops.

    Other nodes attach their own state to the patcher object, e.g. LanPaint does
    ``model.LanPaint_cfg_BIG = cfg`` before sampling and reads it back inside its
    sampler wrapper. ``clone()`` copies the known fields only, so a wrapper that
    clones the model would make those attributes vanish::

        AttributeError: 'ModelPatcherDynamic' object has no attribute 'LanPaint_cfg_BIG'

    Attributes the clone already has are left untouched (only *missing* ones are
    restored), so this cannot clobber ComfyUI's own bookkeeping.
    """
    if src is None or dst is None or src is dst:
        return []
    try:
        items = list(vars(src).items())
    except Exception:
        return []
    copied = []
    for key, value in items:
        if hasattr(dst, key):
            continue
        try:
            setattr(dst, key, value)
            copied.append(key)
        except Exception:
            pass
    return copied


def apply_lllite(model, lllite_name, image, strength, start_percent, end_percent,
                 preserve_wrapper=True, mask=None):
    """Patch ``model`` with the given LLLite and return the patched MODEL."""
    out = get_apply_node().apply(
        model, lllite_name, image, strength, start_percent, end_percent,
        preserve_wrapper, mask,
    )
    # Anima node returns a 1-tuple (MODEL,); tolerate a bare MODEL too.
    patched = out[0] if isinstance(out, (tuple, list)) else out
    kept = _preserve_patcher_attrs(model, patched)
    if kept:
        print(f"{_LOG} model patched; kept custom attrs: {', '.join(kept)}")
    return patched


def resolve_weights_path(lllite_name: str):
    """Absolute path of a controlnet LLLite file, or None."""
    try:
        import folder_paths
    except Exception:
        return None
    try:
        p = folder_paths.get_full_path("controlnet", lllite_name)
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    try:
        for root in folder_paths.get_folder_paths("controlnet"):
            cand = os.path.join(root, lllite_name)
            if os.path.isfile(cand):
                return cand
    except Exception:
        pass
    return None


def read_metadata(lllite_name: str) -> dict:
    """Best-effort LLLite metadata ({} when unavailable)."""
    path = resolve_weights_path(lllite_name)
    if not path:
        return {}
    try:
        mod = load_module()
        fn = getattr(mod, "read_lllite_metadata", None)
        if fn is None:
            return {}
        return dict(fn(path) or {})
    except Exception as e:  # pragma: no cover - diagnostics only
        print(f"{_LOG} read_lllite_metadata failed for {lllite_name}: {e}")
        return {}


def cond_in_channels(lllite_name: str) -> int:
    """3 (RGB) or 4 (RGB+mask) — drives whether we must pass a crop-local mask."""
    try:
        return int(read_metadata(lllite_name).get("lllite.cond_in_channels", 3))
    except Exception:
        return 3


def is_available() -> bool:
    return find_nodes_file() is not None


__all__ = [
    "CANDIDATE_DIRS",
    "custom_nodes_dir",
    "set_base_dir",
    "find_nodes_file",
    "load_module",
    "get_apply_class",
    "get_apply_node",
    "apply_lllite",
    "resolve_weights_path",
    "read_metadata",
    "cond_in_channels",
    "is_available",
]
