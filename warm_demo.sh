#!/usr/bin/env bash
# Warm the demo cache: run the demo floor plan and room video once for real, then confirm repeats are instant.
# Run AFTER the last backend code change + restart: any backend/*.py change re-keys the cache.
# Usage: ./warm_demo.sh [plan.png] [room_video.mp4] [api]
#   defaults: samples/plan1.png  scratch/videos/hotel_room.mp4  http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
PLAN="${1:-samples/plan1.png}"
VIDEO="${2:-scratch/videos/hotel_room.mp4}"
API="${3:-http://localhost:8000}"
[ -f "$PLAN" ] || { echo "no floor plan at $PLAN"; exit 1; }
curl -s --retry 20 --retry-connrefused --retry-delay 1 -o /dev/null "$API/health" || { echo "backend not reachable at $API"; exit 1; }
for i in 1 2; do
  curl -s -o /dev/null -F "file=@$PLAN" "$API/analyze" -w "$(basename "$PLAN")  run $i: HTTP %{http_code}  %{time_total}s\n"
done
if [ -f "$VIDEO" ]; then
  for i in 1 2; do
    curl -s -o /dev/null -F "file=@$VIDEO;type=video/mp4" "$API/scan" -w "$(basename "$VIDEO")  run $i: HTTP %{http_code}  %{time_total}s\n"
  done
else
  echo "no room video at $VIDEO (skipped)"
fi
echo "Run 2 should be ~0.00 s. Agent commands are cached the first time you rehearse them in the browser."
