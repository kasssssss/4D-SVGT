#!/usr/bin/env python3
"""Build the Stage-A manifest and materialize output folders."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import (
    build_manifest,
    materialize_manifest_dirs,
    save_manifest,
    validate_manifest_entries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DVGT-Occ stage-A clip folders.")
    parser.add_argument("--source-root", type=Path, default=Path("../nuscenes_processed_10Hz"))
    parser.add_argument("--output-root", type=Path, default=Path("data/nuscenes_dvgt_v0"))
    parser.add_argument("--manifest", type=Path, default=Path("data/nuscenes_dvgt_v0/manifest_trainval.json"))
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--source-stride", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=40)
    args = parser.parse_args()

    entries = build_manifest(
        source_root=args.source_root,
        output_root=args.output_root,
        split=args.split,
        clip_len=args.clip_len,
        source_stride=args.source_stride,
        clip_stride=args.clip_stride,
    )
    save_manifest(
        entries,
        args.manifest,
        metadata={
            "source_root": str(args.source_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "split": args.split,
            "clip_len": args.clip_len,
            "source_stride": args.source_stride,
            "clip_stride": args.clip_stride,
        },
    )
    materialize_manifest_dirs(entries, args.output_root)
    reports = validate_manifest_entries(entries, args.output_root, strict_outputs=False)
    ok_count = sum(report["ok"] for report in reports)
    print(f"Prepared {len(entries)} clips")
    print(f"Manifest: {args.manifest}")
    print(f"Input validation: {ok_count}/{len(reports)} clips complete")


if __name__ == "__main__":
    main()
