#!/usr/bin/env bash
# On the board (2026-10-29, draft 12): time per image of the medium detector (YOLO11 medium, 1280x1280) over all test
# images, as the det stage of board_p29.sh did for the nano detector (Table VII: medians over all test images), so that
# the two medians share one image set. board_detm.sh timed the first 200 images only, whose ED subset is not
# representative (nano 238 ms there against 274 ms over all ED test images). Outputs runs/p29/det_{ed,td}_m.jsonl.
#   cd ~/pb_edge && setsid nohup bash tools/board_detm_all.sh > runs/p29/detm_all.out 2>&1 < /dev/null &
set -u
H=${PB_EDGE:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge}; K=$H/board_kit; T=$H/tools; O=$H/runs/p29
PY=$H/venv/bin/python
watch() { local tag=$1; shift; $PY $T/board_watch.py run --out $O --tag "$tag" -- "$@"; }
cd $K/detector
for s in ed td; do
  watch det_${s}_m $PY $T/board_det.py detect --model $K/detector/${s}_m_1280x1280 --list $K/detector/images_$s.txt \
    --images $K/detector/images_$s --out $O/det_${s}_m.jsonl --warmup 5 2>&1 | grep -vE "^(W|I) RKNN|^\s*$"
done
echo "DETM_ALL_DONE $(date -u +%H:%M) UTC"
