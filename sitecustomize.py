"""Local Python startup customizations for DVGT training.

This file is loaded automatically when the repo root is on ``PYTHONPATH``.
It provides a small compatibility shim for gsplat against the Torch build
available in the current cluster image.
"""

from __future__ import annotations

import inspect
from pathlib import Path


def _patch_torch_jit_compile() -> None:
    try:
        from torch.utils import cpp_extension as ce
    except Exception:
        return

    original = getattr(ce, "_jit_compile", None)
    if original is None or getattr(original, "_dvgt_compat_patched", False):
        return

    try:
        signature = inspect.signature(original)
    except Exception:
        return

    params = signature.parameters
    if "extra_sycl_cflags" not in params or "with_sycl" not in params:
        return

    def _jit_compile_compat(*args, **kwargs):
        # Older gsplat calls match the pre-SYCL torch signature:
        #   _jit_compile(name, sources, extra_cflags, extra_cuda_cflags,
        #                extra_ldflags, extra_include_paths,
        #                build_directory, verbose, ...)
        # Newer torch inserts ``extra_sycl_cflags`` before ``extra_ldflags``
        # and expects a ``with_sycl`` kwarg.
        if len(args) == 8 and "with_sycl" not in kwargs:
            args = list(args)
            args.insert(4, None)
            kwargs["with_sycl"] = None

        name = args[0] if len(args) > 0 else kwargs.get("name")
        build_directory = args[7] if len(args) > 7 else kwargs.get("build_directory")
        if name == "gsplat_cuda" and build_directory:
            build_dir = Path(build_directory)
            module_path = build_dir / "gsplat_cuda.so"
            if module_path.exists():
                lock_path = build_dir / "lock"
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return ce._import_module_from_library(name, str(build_dir), True)
        return original(*args, **kwargs)

    _jit_compile_compat._dvgt_compat_patched = True  # type: ignore[attr-defined]
    ce._jit_compile = _jit_compile_compat


_patch_torch_jit_compile()
