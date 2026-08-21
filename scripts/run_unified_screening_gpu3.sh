#!/usr/bin/env bash
set -euo pipefail

: "${PRIST_RIS_DATA_ROOT:?set PRIST_RIS_DATA_ROOT}"
: "${PRIOR:?set PRIOR to the existing q0/q3 Ridge artifact}"
: "${SPATIAL_CHECKPOINT:?set SPATIAL_CHECKPOINT to the completed D1+RISCoord checkpoint}"
: "${ANCHOR_CACHE_ROOT:?set ANCHOR_CACHE_ROOT}"

export CUDA_VISIBLE_DEVICES=3
nvidia-smi -i 3
if [[ -n "$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU3 already has a compute PID; refusing to compete."
  exit 1
fi

if [[ ! -f "$ANCHOR_CACHE_ROOT/cache_manifest.json" ]]; then
  prist-ris cache-spatial-anchors \
    --checkpoint "$SPATIAL_CHECKPOINT" \
    --prior "$PRIOR" \
    --data-root "$PRIST_RIS_DATA_ROOT" \
    --device cuda --workers 8 \
    --max-train 4096 --max-validation 1800 \
    --output-root "$ANCHOR_CACHE_ROOT"
fi

prist-ris screen-spatial-modules \
  --prior "$PRIOR" \
  --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8 --physical-gpu-index 3 \
  --execute --confirm-gpu-free

prist-ris screen-temporal \
  --prior "$PRIOR" \
  --spatial-checkpoint "$SPATIAL_CHECKPOINT" \
  --anchor-cache-root "$ANCHOR_CACHE_ROOT" \
  --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8 --physical-gpu-index 3 \
  --include-curvature --execute --confirm-gpu-free
