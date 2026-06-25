#!/usr/bin/env bash
# End-to-end demo: synthesize an L2 dataset, reconstruct it, export USD for Isaac Sim.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-output}"
FRAMES="${2:-24}"

echo "==> [1/2] Full pipeline on synthetic L2 data ($FRAMES frames)"
python3 -m unitree_l2_pipeline.cli run \
    --source synthetic --frames "$FRAMES" --voxel 0.03 --out "$OUT"

echo
echo "==> [2/2] Done. Artifacts in '$OUT/':"
ls -la "$OUT"
echo
echo "Open '$OUT/reconstruction.usda' in Isaac Sim, or run:"
echo "  ./python.sh unitree_l2_pipeline/export/isaac_loader.py --usd $OUT/reconstruction.usda --as-instancer"
