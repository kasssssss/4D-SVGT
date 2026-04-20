#!/usr/bin/env bash
set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT

TAILDIR=data/nuscenes_dvgt_v0/logs/dev_backfill_tailfix_20260419
LOGDIR=data/nuscenes_dvgt_v0/logs/dev_backfill_full_20260419
mkdir -p "$TAILDIR" "$LOGDIR"

while [ ! -f "$TAILDIR/scene009_view0.status" ]; do
  sleep 20
done

python3 - <<'PY'
from pathlib import Path
import json, sys
base = Path('data/nuscenes_dvgt_v0/trainval')
bad = []
for scene in sorted(base.glob('scene_*')):
    for view_id in range(6):
        p = scene / 'sam3_scene' / f'view_{view_id}' / 'progress.json'
        if not p.exists():
            bad.append((scene.name, view_id, 'missing'))
            continue
        obj = json.loads(p.read_text())
        if obj.get('status') != 'done':
            bad.append((scene.name, view_id, obj.get('status')))
if bad:
    print({'not_done': bad}, flush=True)
    sys.exit(2)
print({'all_done': 60}, flush=True)
PY

echo 0 > "$LOGDIR/sam3_backfill_full.status"
bash tools/dvgt_occ/run_sam3_backfill_postprocess.sh
