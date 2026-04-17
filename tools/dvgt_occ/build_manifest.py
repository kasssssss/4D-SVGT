#!/usr/bin/env python3
"""CLI wrapper for ``dvgt_occ.data.manifest``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import _main


if __name__ == "__main__":
    _main()
