#!/usr/bin/env bash
# Download Unitree's real L2 LiDAR recordings (ROS1 bags) used by point_lio_unilidar.
# These are large (the indoor bag is ~520 MB) so they are NOT committed to the repo.
#
# Usage:
#   scripts/fetch_l2_bag.sh [indoor|park|l1] [dest_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-indoor}"
DEST="${2:-data/captures}"
mkdir -p "$DEST"

case "$WHICH" in
  indoor) URL="https://oss-global-cdn.unitree.com/static/L2%20Indoor%20Point%20Cloud%20Data.bag"; NAME="L2_Indoor.bag" ;;
  park)   URL="https://oss-global-cdn.unitree.com/static/L2%20Park%20Point%20Cloud%20Data.bag";   NAME="L2_Park.bag" ;;
  l1)     URL="https://oss-global-cdn.unitree.com/static/unilidar-2023-09-22-12-42-04.zip";        NAME="L1_unilidar.zip" ;;
  *) echo "unknown dataset '$WHICH' (use: indoor | park | l1)"; exit 1 ;;
esac

echo "==> downloading $WHICH -> $DEST/$NAME"
curl -fSL --retry 4 --retry-delay 2 -o "$DEST/$NAME" "$URL"
echo "==> done: $DEST/$NAME"
echo
echo "Inspect and reconstruct it with:"
echo "  unitree-l2 bag-info $DEST/$NAME"
echo "  unitree-l2 run --source $DEST/$NAME --max-frames-cap 60 --out output"
