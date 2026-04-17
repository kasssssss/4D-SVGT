#!/usr/bin/env python3
"""Validate DVGT-Occ manifest inputs and generated stage-A artifacts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import load_manifest, validate_manifest_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DVGT-Occ clip manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--check",
        choices=("inputs", "outputs", "all"),
        default="all",
        help="Validate source inputs only, generated outputs only, or both.",
    )
    parser.add_argument("--strict-outputs", action="store_true")
    args = parser.parse_args()

    root = args.root if args.root is not None else args.manifest.parent
    entries = load_manifest(args.manifest)
    strict_outputs = args.strict_outputs or args.check in {"outputs", "all"}
    reports = validate_manifest_entries(entries, root=root, strict_outputs=strict_outputs)
    if args.check == "outputs":
        for report in reports:
            report["ok"] = not report["missing_outputs"]
    elif args.check == "inputs":
        for report in reports:
            report["ok"] = not report["missing_inputs"]
    ok_count = sum(report["ok"] for report in reports)
    print(f"Validation: {ok_count}/{len(reports)} clips complete")
    for report in reports:
        if not report["ok"]:
            print(report)
            break


if __name__ == "__main__":
    main()
