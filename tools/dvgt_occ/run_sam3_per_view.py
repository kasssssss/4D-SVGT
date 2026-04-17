#!/usr/bin/env python3
"""Deprecated Stage-A entrypoint kept only to prevent stale workflow usage."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "run_sam3_per_view.py is deprecated. Use run_sam3_scene.py to build scene-level "
        "SAM3 assets, then build_sam3_clip_index.py to derive clip supervision."
    )


if __name__ == "__main__":
    main()
