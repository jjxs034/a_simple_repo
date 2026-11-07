#!/usr/bin/env bash
# On the board (2026-10-29, draft 9): time per image of the medium detector (YOLO11 medium, 1280x1280, the detector of
# the 4B configuration) for the tier selection of Section III-F. Waits until the running P29 chain (mmboard, then
# mmdarkboard) has ended, then runs board_det.py detect over the first 200 test images of each scenario through
# board_watch.py, exactly as the det stage of board_p29.sh does for the nano detector, so that its median is comparable
# with the detector time of Table VII. Outputs: runs/p29/det_{ed,td}_m200.jsonl and their watch records.
#   cd ~/pb_edge && setsid nohup bash tools/board_detm.sh <pid of the chain> > runs/p29/detm.out 2>&1 < /dev/null &
set -u
H=${PB_EDGE:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge}; K=$H/board_kit; T=$H/tools; O=$H/runs/p29
PY=$H/venv/bin/python
watch() { local tag=$1; shift; $PY $T/board_watch.py run --out $O --tag "$tag" -- "$@"; }
CHAIN=${1:?pid of the running P29 chain}
echo "waiting for pid $CHAIN since $(date -u +%H:%M) UTC"
while kill -0 "$CHAIN" 2>/dev/null; do sleep 60; done
echo "chain ended at $(date -u +%H:%M) UTC"
cd $K/detector
for s in ed td; do
  head -n 200 images_$s.txt > $O/images_${s}_200.txt
  watch det_${s}_m200 $PY $T/board_det.py detect --model $K/detector/${s}_m_1280x1280 --list $O/images_${s}_200.txt \
    --images $K/detector/images_$s --out $O/det_${s}_m200.jsonl --warmup 5 2>&1 | grep -vE "^(W|I) RKNN|^\s*$"
done
echo "DETM_DONE $(date -u +%H:%M) UTC"
