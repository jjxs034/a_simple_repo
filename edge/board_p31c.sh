#!/usr/bin/env bash
# On the board: supplement 3 of protocol P31 (docs/results/round2_protocol.md), the two memory clocks that P31 measured
# only at NPU 1 GHz, swept over the NPU clock: DDR 1,968 MHz at NPU 300-700 and DDR 534 MHz at NPU 300, 500, 700. Per
# point the synthetic lengths of item 1(a) and the 256-token vision encoder (the thumbnail of the board). Same watched
# runs as board_p31b.sh (an invalid run is set aside as *_void<n> and repeated after cooling, at most twice); outputs go
# to runs/p31c/.
#   bash board_p31c.sh
set -u
H=${PB_EDGE:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge}; K=$H/board_kit; P=$H/board_p29; T=$H/tools; O=$H/runs/p31c; R=$H/runs/p31
PY=$H/venv/bin/python
MODEL=$P/model/qwen3vl2b_f_listn2b_s0_w8a8_opt0_rk3588.rkllm
V256=$H/board_p29v/qwen3vl2b_vision_512x512_rk3588.rknn
mkdir -p $O
setf() { bash $T/board_p31.sh set "$1" "$2"; }

measure() {  # measure TAG OUTFILE CMD...: as in board_p31.sh
  local tag=$1 out=$2; shift 2
  for try in 1 2 3; do
    $PY $T/board_watch.py run --out $O --tag "$tag" -- "$@" && return 0
    [ $try = 3 ] && { echo "VOID_GIVEUP $tag"; return 1; }
    for f in "$out" $O/watch_$tag.csv $O/watch_$tag.json; do [ -e "$f" ] && mv "$f" "${f%.*}_void$try.${f##*.}"; done
    echo "VOID $tag (attempt $try), cooling 120 s"; sleep 120
  done
}

for point in 1968:300 1968:400 1968:500 1968:600 1968:700 534:300 534:500 534:700; do
  d=${point%%:*}; f=${point##*:}
  setf $f $d || exit 1
  measure len_npu${f}_ddr$d $O/len_npu${f}_ddr$d.jsonl $PY $T/board_bench.py --lib $K/librkllmrt.so --model $MODEL \
    --prompts $R/prompts_len5.jsonl --out $O/len_npu${f}_ddr$d.jsonl --template $P/tok/template.json --warmup 5
  measure npu${f}_ddr${d}_v256 $O/npu${f}_ddr${d}_v256.log bash -c "$PY $T/board_det.py time --model $V256 --shape 512 512 --runs 3 --warmup 1 --cores 0_1_2 2>&1 | grep -vE '^(W|I) RKNN|^\s*$' | tee $O/npu${f}_ddr${d}_v256.log"
done
setf 1000 2400; echo P31C_DONE
